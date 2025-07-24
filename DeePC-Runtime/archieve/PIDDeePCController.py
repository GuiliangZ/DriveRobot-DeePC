#!/usr/bin/env python3
"""
DeePC - Data-Enabled Predictive Control controller use data directly in the optimization framework
to track the pre-defined speed profile precisely.
Runs at 100 Hz (T_s = 0.01 s) to track a reference speed profile, reading v_meas from CAN,
and writing a duty‐cycle (–15% to +100%) to a PCA9685 PWM board.

Required setup:
  pip install numpy scipy cantools python-can adafruit-circuitpython-pca9685
  sudo pip install Jetson.GPIO
  # !!!!!!!!!! Always run this command line in the terminal to start the CAN reading: !!!!!!!!!
# CAN setup
sudo modprobe can       # core CAN support
sudo modprobe can_raw   # raw-socket protocol
sudo modprobe can_dev   # network interface “canX” support
sudo modprobe kvaser_usb
sudo ip link set can0 down                         # if it was already up
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 500000
sudo ip link set can1 up
    i2cdetect -l
    ls /sys/class/net/ | grep can
    """
import psutil, os
import time
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd
from collections import deque

# import Jetson.GPIO as GPIO
# GPIO.cleanup()
# import board
# import busio
# from adafruit_pca9685 import PCA9685

import cantools
import can
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver
import deepctools as dpc
from deepctools.util import *
from utils_deepc import *
from smbus2 import SMBus
import DeePCAcados as dpcAcados
from DeePCCVXPYSolver import DeePCCVXPYWrapper
from DeePCParameterTuner import DeePCParameterTuner, DeePCParameters
from OnlineParameterTuner import OnlineParameterTuner, create_online_tuner
from AdaptiveOnlineTuner import AdaptiveOnlineTuner, create_adaptive_tuner, AdaptiveTuningConfig
from deepc_config import *

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
algorithm_name = "PID_DeePC_10Hz_Comp"
latest_speed = None                                 # Measured speed (kph) from CAN
latest_force = None                                 # Measured force (N) from CAN (unused here)
dyno_can_running  = True                            # Flag to stop the CAN thread on shutdown
veh_can_running  = True 
BMS_socMin = None                                   # Measured current vehicle SOC from Vehicle CAN
# dyno_can_running  = False                           # For temperal debugging
# veh_can_running  = False 
CP2112_BUS   = 3         # e.g. /dev/i2c-3

# ──────────────────────────── CAN LISTENER THREAD ──────────────────────────────
def dyno_can_listener_thread(dbc_path: str, can_iface: str):
    """
    Runs in a background thread. Listens for 'Speed_and_Force' on can_iface,
    decodes it using KAVL_V3.dbc, and updates globals latest_speed & latest_force.
    """
    global latest_speed, latest_force, dyno_can_running

    try:
        db = cantools.database.load_file(dbc_path)
    except FileNotFoundError:
        print(f"[CAN⋅Thread] ERROR: Cannot find DBC at '{dbc_path}'. Exiting CAN thread.")
        return

    try:
        speed_force_msg = db.get_message_by_name('Speed_and_Force')
    except KeyError:
        print("[CAN⋅Thread] ERROR: 'Speed_and_Force' not found in DBC. Exiting CAN thread.")
        return

    try:
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
    except OSError:
        print(f"[CAN⋅Thread] ERROR: Cannot open CAN interface '{can_iface}'. Exiting CAN thread.")
        return

    print(f"[CAN⋅Thread] Listening on {can_iface} for ID=0x{speed_force_msg.frame_id:03X}…")
    while dyno_can_running:
        msg = bus.recv(timeout=1.0)
        if msg is None:
            continue
        if msg.arbitration_id != speed_force_msg.frame_id:
            continue
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
        except KeyError:
            continue

        s = decoded.get('Speed_kph')
        f = decoded.get('Force_N')
        if s is not None:
            if -0.1 < s < 0.1:
                latest_speed = int(round(s))
            else:
                latest_speed = float(s)
        if f is not None:
            latest_force = float(f)

    bus.shutdown()
    print("[CAN⋅Thread] Exiting CAN thread.")

def veh_can_listener_thread(dbc_path: str, can_iface: str):
    """
    Runs in a background thread. Listens for 'BMS_socMin' on can_iface,
    decodes it using vehBus.dbc, and updates globals BMS_socMin.
    """
    global BMS_socMin, veh_can_running

    try:
        db = cantools.database.load_file(dbc_path)
    except FileNotFoundError:
        print(f"[CAN⋅Thread] ERROR: Cannot find DBC at '{dbc_path}'. Exiting CAN thread.")
        return

    try:
        bms_soc_msg = db.get_message_by_name('BMS_socStatus')
    except KeyError:
        print("[CAN⋅Thread] ERROR: 'BMS_socStatus' not found in DBC. Exiting CAN thread.")
        return

    try:
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
    except OSError:
        print(f"[CAN⋅Thread] ERROR: Cannot open CAN interface '{can_iface}'. Exiting CAN thread.")
        return

    print(f"[CAN⋅Thread] Listening on {can_iface} for ID=0x{bms_soc_msg.frame_id:03X}…")
    while veh_can_running:
        msg = bus.recv(timeout=3.0)
        if msg is None:
            continue
        if msg.arbitration_id != bms_soc_msg.frame_id:
            continue
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
        except KeyError:
            continue

        BMS_socMin = decoded.get('BMS_socMin')
        if BMS_socMin is not None:
            if BMS_socMin <= 2:
                BMS_socMin = int(round(BMS_socMin))
            else:
                BMS_socMin = float(BMS_socMin)
    bus.shutdown()
    print("[CAN⋅Thread] Exiting CAN thread.")

# ─────────────────────────────── MAIN CONTROL ─────────────────────────────────
if __name__ == "__main__":
    # System parameters
    max_delta = 80.0                                    # maximum % change per 0.01 s(Ts) tick - regulate the rate of change of pwm output u
    SOC_CycleStarting = 0.0                             # Managing Vehicle SOC - record SOC at every starting point of the cycle
    SOC_Stop = MIN_SOC_STOP                            # Stop the test at minimum SOC from configuration
    FeedFwdTime = 0.65                                  # feedforward reference speed time
    T_f = 5000.0                                        # Derivative filter coefficient. Formula uses: D_f[k] = D_f[k-1] + (T_s / (T_s + T_f)) * (D_k - D_f[k-1])

    # ─── LOAD CONFIGURATION SETTINGS ───────────────────────────────────────────────
    # Validate configuration before proceeding
    config_errors = validate_config()
    if config_errors:
        print("[ERROR] Configuration validation failed:")
        for error in config_errors:
            print(f"  - {error}")
        print("Please fix configuration in deepc_config.py and restart")
        exit(1)
    
    # Use configuration values
    Ts = 1.0 / CONTROL_FREQUENCY                    # Control sampling time
    u_dim = 1                                       # SISO system - control input dimension
    y_dim = 1                                       # SISO system - output dimension
    
    # Load DeePC parameters from configuration
    deepc_params = MANUAL_DEEPC_PARAMS
    
    # Extract parameters for backward compatibility
    Tini = deepc_params.Tini
    THorizon = deepc_params.THorizon
    hankel_subB_size = deepc_params.hankel_subB_size
    Q_val = deepc_params.Q_val
    R_val = deepc_params.R_val
    lambda_g_val = deepc_params.lambda_g_val
    lambda_y_val = deepc_params.lambda_y_val
    lambda_u_val = deepc_params.lambda_u_val
    T = hankel_subB_size
    g_dim = T - Tini - THorizon + 1
    
    print(f"[Main] DeePC Controller Configuration:")
    print(f"  Control frequency: {CONTROL_FREQUENCY} Hz (Ts = {Ts:.3f}s)")
    print(f"  Solver: {SOLVER_TYPE}")
    print(f"  Parameter tuning: {'Enabled' if ENABLE_PARAMETER_TUNING else 'Disabled'}")
    print(f"  DeePC structure: Tini={Tini}, THorizon={THorizon}, Hankel size={hankel_subB_size}")
    print(f"  Cost weights: Q={Q_val}, R={R_val}")
    print(f"  Regularization: λg={lambda_g_val}, λy={lambda_y_val}, λu={lambda_u_val}")
    
    # Create weighting matrices with time-varying weights for better performance
    decay_rate_q = deepc_params.decay_rate_q
    decay_rate_r = deepc_params.decay_rate_r
    
    # Define Q and R with exponential decay (focus more on near-term predictions)
    q_weights = Q_val * np.exp(-decay_rate_q * np.arange(THorizon))
    r_weights = R_val * np.exp(-decay_rate_r * np.arange(THorizon))
    Q = np.diag(q_weights)                              # Time-varying output tracking weights
    R = np.diag(r_weights)                              # Time-varying control effort weights
    
    # Regularization matrices for robust DeePC
    lambda_g = np.diag(np.tile(lambda_g_val, g_dim))    # g-vector regularization
    lambda_y = np.diag(np.tile(lambda_y_val, Tini))     # Output mismatch regularization  
    lambda_u = np.diag(np.tile(lambda_u_val, Tini))     # Input mismatch regularization
    
    print(f"  Time-varying weights: Q_decay={decay_rate_q}, R_decay={decay_rate_r}")
    print(f"  Degrees of freedom: g_dim={g_dim}")
    
    # Check for potential dimension issues
    if g_dim <= 0:
        print(f"[ERROR] Invalid g_dim={g_dim}. Check parameter configuration.")
        print(f"  Current: Tini={Tini}, THorizon={THorizon}, hankel_subB_size={hankel_subB_size}")
        print(f"  g_dim = hankel_subB_size - Tini - THorizon + 1 = {hankel_subB_size} - {Tini} - {THorizon} + 1 = {g_dim}")
        exit(1)
    
    # Performance monitoring
    solve_time_history = []
    success_rate_history = []
    
    # Online parameter tuning setup
    online_tuner = None
    if ENABLE_ONLINE_TUNING:
        if ONLINE_TUNING_MODE == "adaptive":
            # Create adaptive tuner with custom configuration
            config = AdaptiveTuningConfig(
                speed_bins=ADAPTIVE_SPEED_BINS,
                min_samples_per_condition=ADAPTIVE_MIN_SAMPLES,
                max_samples_per_condition=ADAPTIVE_MAX_SAMPLES,
                similarity_threshold=ADAPTIVE_SIMILARITY_THRESHOLD,
                accel_threshold=ADAPTIVE_ACCEL_THRESHOLD
            )
            online_tuner = AdaptiveOnlineTuner(config, deepc_params)
            print(f"[Main] Adaptive online parameter tuning enabled:")
            print(f"  Start delay: {ONLINE_TUNING_START_DELAY} cycles")
            print(f"  Speed bins: {ADAPTIVE_SPEED_BINS}")
            print(f"  Min samples per condition: {ADAPTIVE_MIN_SAMPLES}")
        else:
            # Use basic online tuner
            online_tuner = create_online_tuner(deepc_params)
            print(f"[Main] Basic online parameter tuning enabled:")
            print(f"  Start delay: {ONLINE_TUNING_START_DELAY} cycles")
            print(f"  Evaluation window: {EVALUATION_WINDOW_SIZE} cycles")
        print(f"  Update interval: {PARAMETER_UPDATE_INTERVAL} cycles")

    # ─── PARAMETER TUNING (if enabled) ──────────────────────────────────────────────
    PROJECT_DIR = Path(__file__).resolve().parent 
    DATA_DIR = PROJECT_DIR / "dataForHankle" / "smallDataSet"
    
    if ENABLE_PARAMETER_TUNING:
        print(f"[Main] Starting DeePC parameter tuning with {TUNING_METHOD}...")
        tuner = DeePCParameterTuner(str(DATA_DIR), target_frequency=CONTROL_FREQUENCY)
        
        # Run parameter optimization
        best_params, best_metrics = tuner.optimize_parameters(
            method=TUNING_METHOD,
            max_evaluations=MAX_TUNING_EVALUATIONS,
            initial_params=deepc_params
        )
        
        # Update parameters with tuned values
        deepc_params = best_params
        Tini = deepc_params.Tini
        THorizon = deepc_params.THorizon
        hankel_subB_size = deepc_params.hankel_subB_size
        Q_val = deepc_params.Q_val
        R_val = deepc_params.R_val
        lambda_g_val = deepc_params.lambda_g_val
        lambda_y_val = deepc_params.lambda_y_val
        lambda_u_val = deepc_params.lambda_u_val
        T = hankel_subB_size
        g_dim = T - Tini - THorizon + 1
        
        # Recompute weighting matrices with new parameters
        decay_rate_q = deepc_params.decay_rate_q
        decay_rate_r = deepc_params.decay_rate_r
        q_weights = Q_val * np.exp(-decay_rate_q * np.arange(THorizon))
        r_weights = R_val * np.exp(-decay_rate_r * np.arange(THorizon))
        Q = np.diag(q_weights)
        R = np.diag(r_weights)
        lambda_g = np.diag(np.tile(lambda_g_val, g_dim))
        lambda_y = np.diag(np.tile(lambda_y_val, Tini))
        lambda_u = np.diag(np.tile(lambda_u_val, Tini))
        
        # Generate and save tuning report
        report = tuner.generate_parameter_recommendations(best_params, best_metrics)
        print(report)
        
        tuning_results_file = PROJECT_DIR / f"deepc_tuning_results_{int(time.time())}.json"
        tuner.save_tuning_results(str(tuning_results_file), best_params, best_metrics)
        
        print(f"[Main] Parameter tuning completed. Optimized parameters will be used.")
        print(f"[Main] Tuning results saved to {tuning_results_file}")
        
        # Force solver recompilation with new parameters
        recompile_solver = False
    else:
        # Always recompile if dimensions have changed
        recompile_solver = False  # Force recompilation to avoid dimension mismatch
    
    # ─── DeePC SOLVER SETUP ─────────────────────────────────────────────────────
    print(f"[Main] Initializing DeePC solver: {SOLVER_TYPE}")
    CACHE_FILE_Ori_DATA = os.path.join(DATA_DIR, "hankel_dataset.npz")          # Cache the previously saved SISO data
    CACHE_FILE_HANKEL_DATA = os.path.join(DATA_DIR, "hankel_matrix.npz")        # Cache the previously saved Hankel matrix
    if os.path.isfile(CACHE_FILE_Ori_DATA) and USE_CACHED_HANKEL_DATA:
        print(f"[Main] Using cached input output data from {CACHE_FILE_Ori_DATA}")
        npz = np.load(CACHE_FILE_Ori_DATA, allow_pickle=True)
        ud, yd = npz['ud'], npz['yd']
    else:
        print("[Main] Start to load the fresh offline data for building hankel matrix... this may take a while")
        ud, yd = load_timeseries(DATA_DIR)          # history data collected offline to construct Hankel matrix; size (T, ud/yd)
        np.savez(CACHE_FILE_Ori_DATA, ud=ud, yd=yd)
        print(f"[Main] Finished loading data for hankel matrix, and saved to {CACHE_FILE_Ori_DATA}")

    #DeePC_kickIn_time = 100                                                      # because we need to build hankel matrix around the current time point, should be half of the hankel_subB_size
    if os.path.isfile(CACHE_FILE_HANKEL_DATA) and USE_CACHED_HANKEL_DATA:
        print(f"[Main] Using cached hankel matrix data from {CACHE_FILE_HANKEL_DATA}")
        npz_hankel = np.load(CACHE_FILE_HANKEL_DATA)
        Up, Uf, Yp, Yf = npz_hankel['Up'], npz_hankel['Uf'], npz_hankel['Yp'], npz_hankel['Yf']
        # print(f"Up_cur shape{Up.shape} value: {Up}, "
        # f"Uf_cur shape{Uf.shape} value: {Uf}, "
        # f"Yp_cur shape{Yp.shape} value: {Yp}, "
        # f"Yf_cur shape{Uf.shape} value: {Uf}, ")
    else:
        print("[Main] Start to make hankel matrix data from cache... this may take a while")
        Up, Uf, Yp, Yf = hankel_full(ud, yd, Tini, THorizon)
        np.savez(CACHE_FILE_HANKEL_DATA, Up=Up, Uf=Uf, Yp=Yp, Yf=Yf)
        print(f"[Main] Finished making data for hankel matrix with shape Up{Up.shape}, Uf{Uf.shape}, Yp{Yp.shape}, Yf{Yf.shape}, and saved to {CACHE_FILE_HANKEL_DATA}")

    # Define constraints based on configuration
    ineqconidx = {'u': [0], 'y': [0]}
    ineqconbd = {
        'lbu': np.array([U_MIN]), 
        'ubu': np.array([U_MAX]),
        'lby': np.array([Y_MIN]), 
        'uby': np.array([Y_MAX])
    }
    
    # Add rate constraints if enabled
    if ENABLE_RATE_LIMITING:
        max_delta_per_cycle = MAX_CONTROL_RATE * Ts  # Convert rate limit to per-cycle limit
        ineqconidx['du'] = [0]
        ineqconbd['lbdu'] = np.array([-max_delta_per_cycle])
        ineqconbd['ubdu'] = np.array([max_delta_per_cycle])
        print(f"[Main] Rate limiting enabled: max change = ±{max_delta_per_cycle:.1f}% per cycle")             

    # Initialize solver based on selected type
    dpc_args = [u_dim, y_dim, T, Tini, THorizon]
    dpc_kwargs = dict(ineqconidx=ineqconidx, ineqconbd=ineqconbd)
    
    if SOLVER_TYPE == "acados":
        # Clean up old Acados compilation if recompiling
        if recompile_solver:
            import shutil
            acados_dirs = ["c_generated_code", "__pycache__"]
            for dir_name in acados_dirs:
                if os.path.exists(dir_name):
                    try:
                        shutil.rmtree(dir_name)
                        print(f"[Main] Cleaned up old {dir_name} directory")
                    except Exception as e:
                        print(f"[Main] Warning: Could not clean {dir_name}: {e}")
            
            # Remove old JSON file
            json_file = "DeePC_acados_ocp.json"
            if os.path.exists(json_file):
                try:
                    os.remove(json_file)
                    print(f"[Main] Removed old {json_file}")
                except Exception as e:
                    print(f"[Main] Warning: Could not remove {json_file}: {e}")
        
        dpc = dpcAcados.deepctools(*dpc_args, **dpc_kwargs)
        dpc.init_DeePCAcadosSolver(recompile_solver=recompile_solver, ineqconidx=ineqconidx, ineqconbd=ineqconbd)
        print(f"[Main] Using Acados solver for real-time DeePC (g_dim={g_dim})")
    elif SOLVER_TYPE == "cvxpy":
        dpc = DeePCCVXPYWrapper(*dpc_args, **dpc_kwargs)
        dpc.init_DeePCAcadosSolver(recompile_solver=recompile_solver, ineqconidx=ineqconidx, ineqconbd=ineqconbd)
        print(f"[Main] Using CVXPY solver for DeePC (backup option)")
    else:
        raise ValueError(f"Unsupported solver type: {SOLVER_TYPE}")
    dpc_opts = {                            # cs.nlpsol solver parameters - not used in acados
        'ipopt.max_iter': 100,  # 50
        'ipopt.tol': 1e-5,
        'ipopt.print_level': 1,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6,
    }
    # Specify what solver wanted to use - # Those solver are available as part of the deepctools, but may be slower than DeePCAcados for real time application
    # dpc.init_DeePCsolver(uloss='u', ineqconidx=ineqconidx, ineqconbd=ineqconbd, opts=dpc_opts)            
    # dpc.init_RDeePCsolver(uloss='u', ineqconidx=ineqconidx, ineqconbd=ineqconbd, opts=dpc_opts)
    # dpc.init_FullRDeePCsolver(uloss='u', ineqconidx=ineqconidx, ineqconbd=ineqconbd, opts=dpc_opts)
    print("[Main] Finished compiling DeePC problem, starting the nominal system setup procedure!")
    
    # ─── PCA9685 PWM SETUP ──────────────────────────────────────────────────────
    bus = SMBus(CP2112_BUS)

    # ─── START CAN LISTENER THREAD ───────────────────────────────────────────────
    DYNO_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/KAVL_V3.dbc'
    DYNO_CAN_INTERFACE = 'can0'
    VEH_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/vehBus.dbc'
    VEH_CAN_INTERFACE = 'can1'
    if dyno_can_running:
        dyno_can_thread = threading.Thread(
            target=dyno_can_listener_thread,
            args=(DYNO_DBC_PATH, DYNO_CAN_INTERFACE),
            daemon=True
        )
        dyno_can_thread.start()
    if veh_can_running:
        veh_can_thread = threading.Thread(
            target=veh_can_listener_thread,
            args=(VEH_DBC_PATH, VEH_CAN_INTERFACE),
            daemon=True
        )
        veh_can_thread.start()

    # ─── System Setup ────────────────────────────────────────────────
    base_folder = ""
    all_cycles = load_drivecycle_mat_files(base_folder) # Load reference cycle from .mat(s)
    cycle_keys = choose_cycle_key(all_cycles)           # Prompt the user to choose multiple drive cycles the user wish to test
    veh_modelName = choose_vehicleModelName()           # Prompt the user to choose the model of testing vehicle for logging purpose

    for idx, cycle_key in enumerate(cycle_keys):
        # ----------------Stop the test if the vehicle SOC is too low to prevent draining the vehicle---------------------
        if BMS_socMin is not None and BMS_socMin <= SOC_Stop:
            break
        else:
            SOC_CycleStarting = BMS_socMin
            if SOC_CycleStarting is not None:
                SOC_CycleStarting = round(SOC_CycleStarting, 2)
            else:
                SOC_CycleStarting = 0.0

        # ----------------Loading current cycle data----------------------------------------------------------------------
        cycle_data = all_cycles[cycle_key]
        print(f"\n[Main] Using reference cycle '{cycle_key}'")
        mat_vars = [k for k in cycle_data.keys() if not k.startswith("__")]
        if len(mat_vars) != 1:
            raise RuntimeError(f"Expected exactly one variable in '{cycle_key}', found {mat_vars}")
        varname = mat_vars[0]
        ref_array = cycle_data[varname]
        if ref_array.ndim != 2 or ref_array.shape[1] < 2:
            raise RuntimeError(f"Expected '{varname}' to be N×2 array. Got shape {ref_array.shape}")

        # -----------------Extract reference time (s) and speed (mph)--------------------------------------------------------
        ref_time  = ref_array[:, 0].astype(float).flatten()
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()         # All the drive cycle .mat data file speed are in MPH
        ref_speed = ref_speed_mph * 1.60934                             # now in kph
        print(f"[Main] Reference loaded: shape = {ref_array.shape}")
        ref_horizon_speed = ref_speed[:THorizon].reshape(-1,1)          # Prepare reference speed horizon for DeePC - Length 

        # -----------------Reset states--------------------------------------------------------------------------------------
        loop_count     = 0
        hankel_idx     = 0
        prev_error     = 0.0
        t_deepc        = 0.0
        I_state        = 0.0
        D_f_state      = 0.0
        u_prev         = 0.0
        g_prev         = None                                           # Record DeePC decision matrix g for solver hot start   
        prev_ref_speed = None                                           # Track previous reference speed 
        exist_feasible_sol = False
        PID_control_activate = True
        # Initialize DeePC solver output variables
        u_opt = np.zeros(THorizon)
        g_opt = np.zeros(g_dim)
        cost = 0.0
        
        # Online tuning variables
        tuning_started = False
        last_param_update_cycle = 0
        u_history      = deque([0.0]*Tini,maxlen=Tini)                  # Record the history of control input for DeePC generating u_ini
        spd_history    = deque([0.0]*Tini,maxlen=Tini)                  # Record the history of control input for DeePC generating y_ini
        u_init = np.array(u_history).reshape(-1, 1)                     # shape (u_dim*Tini,1)
        y_init = np.array(spd_history).reshape(-1, 1)
        log_data       = []                                             # Prepare logging
        # Record loop‐start time so we can log elapsed time from 0.0
        next_time      = time.perf_counter()
        t0             = time.perf_counter()
        print(f"\n[Main] Starting cycle '{cycle_key}' on {veh_modelName}, duration={ref_time[-1]:.2f}s")

        # ----------------Real-time Effort (try to avoid system lags - each loop more than 10ms)
        # For real-time effort - put into kernel - linux 5.15.0-1087-realtime for strict time update - but this kernel doesn't have wifi and nvidia drive
        SCHED_FIFO = os.SCHED_FIFO
        priority = 99
        param = os.sched_param(priority)
        try:
            os.sched_setscheduler(0, SCHED_FIFO, param)
            print(f"[Main] Real-time scheduling enabled: FIFO, priority={priority}")
        except:
            print("Need to run as root (or have CAP_SYS_NICE)")

    # ─── MAIN 10 Hz CONTROL LOOP ─────────────────────────────────────────────────
        print("[Main] Entering 10 Hz control loop. Press Ctrl+C to exit.\n")
        try:
            while True:
                loop_start = time.perf_counter()
                sleep_for = next_time - loop_start
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_time = time.perf_counter()
                elapsed_time = loop_start - t0                          # Compute elapsed time since loop start

                # ── Interpolate reference speed at t and t+Ts ───────────────────
                if elapsed_time <= ref_time[0]:
                    rspd_now = ref_speed[0]
                elif elapsed_time >= ref_time[-1]:
                    rspd_now = 0.0
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))

                # -- Interpolate reference speed for DeePC ref_horizon_speed -------------------------
                t_future = elapsed_time + Ts * np.arange(THorizon)      # look 0.01 * THorizon s ahead of time
                if t_future[-1] >= ref_time[-1]:                        # if the last future time is beyond your reference horizon...
                    valid_mask = t_future <= ref_time[-1]               # build a boolean mask of all valid future times
                    THorizon = int(valid_mask.sum())                    # shrink THorizon to only those valid steps - !! Horizon will change in last few steps
                    t_future = t_future[valid_mask]             
                ref_horizon_speed = np.interp(t_future, ref_time, ref_speed)
                ref_horizon_speed = ref_horizon_speed.reshape(-1, 1)
                
                # ── Compute current error e[k] and future error e_fut[k] ────────
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                e_k    = rspd_now - v_meas
                
                # ── DEEPC CONTROL IMPLEMENTATION ────────────────────────────────────
                # Get current Hankel sub-matrices
                Up_cur, Uf_cur, Yp_cur, Yf_cur = hankel_subBlocks(Up, Uf, Yp, Yf, Tini, THorizon, hankel_subB_size, hankel_idx)
                
                # Check data validity
                arrays_to_check = [Up_cur, Uf_cur, Yp_cur, Yf_cur, u_init, y_init, ref_horizon_speed]
                DeePC_data_valid = all(np.any(arr != 0) for arr in arrays_to_check) and all(arr is not None for arr in arrays_to_check)
                
                # Attempt DeePC solve
                if DeePC_data_valid:
                    try:
                        u_opt, g_opt, t_deepc, exist_feasible_sol, cost = dpc.acados_solver_step(
                            uini=u_init, yini=y_init, yref=ref_horizon_speed,
                            Up_cur=Up_cur, Uf_cur=Uf_cur, Yp_cur=Yp_cur, Yf_cur=Yf_cur, 
                            Q_val=Q, R_val=R, lambda_g_val=lambda_g, 
                            lambda_y_val=lambda_y, lambda_u_val=lambda_u, g_prev=g_prev
                        )
                    except RuntimeError as e:
                        if "mismatching dimension" in str(e):
                            print(f"[ERROR] Dimension mismatch in Acados solver: {e}")
                            print(f"[ERROR] Current g_dim={g_dim}, but solver expects different dimension")
                            print(f"[ERROR] Try deleting c_generated_code directory and restart")
                            print(f"[ERROR] Falling back to PID control")
                        else:
                            print(f"[ERROR] Acados solver error: {e}")
                        
                        # Set defaults for error case
                        u_opt, g_opt, t_deepc, exist_feasible_sol, cost = (
                            np.zeros(THorizon), np.zeros(g_dim), 0.0, False, float('inf')
                        )
                    
                    # Track solver performance
                    solve_time_history.append(t_deepc)
                    success_rate_history.append(exist_feasible_sol)
                    
                    # Keep track of last 100 solves for performance monitoring
                    if len(solve_time_history) > 100:
                        solve_time_history.pop(0)
                        success_rate_history.pop(0)
                    
                    # Update warm start for next iteration
                    if exist_feasible_sol and not np.all(u_opt == 0):
                        g_prev = g_opt
                    else:
                        g_prev = None
                    
                    # Determine if DeePC solution is acceptable
                    max_allowed_solve_time = (Ts * 1000 * MAX_SOLVE_TIME_RATIO)  # Convert to ms
                    
                    DeePC_control = (DeePC_data_valid and 
                                   exist_feasible_sol and 
                                   cost >= 0 and 
                                   t_deepc < max_allowed_solve_time)
                    
                    # Adaptive fallback if performance degrades
                    if ENABLE_ADAPTIVE_FALLBACK and len(solve_time_history) >= 10:
                        recent_avg_time = np.mean(solve_time_history[-10:])
                        recent_success_rate = np.mean(success_rate_history[-10:])
                        
                        if recent_avg_time > max_allowed_solve_time or recent_success_rate < 0.8:
                            DeePC_control = False
                            print(f"[Control] Adaptive fallback to PID: solve_time={recent_avg_time:.1f}ms, success_rate={recent_success_rate:.2f}")
                else:
                    # No valid data available
                    DeePC_control = False
                    t_deepc = 0.0
                    cost = float('inf')
                    exist_feasible_sol = False
                    u_opt = np.zeros(THorizon)
                    g_opt = np.zeros(g_dim)

                # Apply control decision
                if DeePC_control and exist_feasible_sol:
                    u_unclamped = u_opt[0]
                    PID_control_activate = False
                else:
                    # Fallback to PID controller
                    u_PID, P_term, I_out, D_term, e_k_mph = compute_pid_control(
                        elapsed_time, FeedFwdTime, ref_time, ref_speed, v_meas, e_k, 
                        prev_error, I_state, D_f_state, Ts, T_f
                    )
                    u_unclamped = u_PID
                    PID_control_activate = True

                # ── SAFETY LIMITS AND CONTROL OUTPUT PROCESSING ────────────────────────
                # Apply control limits from configuration
                u = float(np.clip(u_unclamped, U_MIN, U_MAX))
                
                # Rate limiting (additional safety layer)
                if ENABLE_RATE_LIMITING:
                    max_delta_safety = MAX_CONTROL_RATE * Ts * 1.2  # 20% safety margin
                    lower_bound = u_prev - max_delta_safety
                    upper_bound = u_prev + max_delta_safety
                    u = float(np.clip(u, lower_bound, upper_bound))
                
                # Emergency speed limit safety
                if latest_speed is not None and latest_speed >= EMERGENCY_SPEED_LIMIT:
                    u = 0.0
                    print(f"[SAFETY] Emergency stop: speed {latest_speed:.1f} kph >= limit {EMERGENCY_SPEED_LIMIT:.1f} kph")
                    break

                # ──  Send PWM to PCA9685: accel (ch=0) if u>=0, else brake (ch=4) ──
                if u >= 0.0:
                    set_duty_cycle(bus, 4, 0.0)                                    # ensure brake channel is zero
                    set_duty_cycle(bus, 0, u)                                      # channel 0 = accelerator
                else:
                    set_duty_cycle(bus, 0, 0.0)                                    # ensure brake channel is zero
                    set_duty_cycle(bus, 4, -u)                                     # channel 4 = brake

                actual_elapsed_time = round((time.perf_counter() - loop_start)*1000, 3)
                actual_control_frequency = 1/(actual_elapsed_time / 1000) if actual_elapsed_time > 0 else 0.0
                
                # ── PERFORMANCE MONITORING AND LOGGING ─────────────────────────────
                # Print performance stats periodically
                if LOG_SOLVER_PERFORMANCE and loop_count % PERFORMANCE_STATS_INTERVAL == 0 and len(solve_time_history) > 0:
                    avg_solve_time = np.mean(solve_time_history)
                    avg_success_rate = np.mean(success_rate_history)
                    print(f"[Performance] Last {len(solve_time_history)} solves: avg_time={avg_solve_time:.2f}ms, success_rate={avg_success_rate:.1%}")
                
                # ── ONLINE PARAMETER TUNING ────────────────────────────────────────
                if online_tuner is not None:
                    # Start tuning after delay period
                    if not tuning_started and loop_count >= ONLINE_TUNING_START_DELAY:
                        online_tuner.start_tuning()
                        tuning_started = True
                        print(f"[OnlineTuning] Started at cycle {loop_count}")
                    
                    # Record performance data
                    if tuning_started:
                        if isinstance(online_tuner, AdaptiveOnlineTuner):
                            # Adaptive tuner needs reference speed and measured speed
                            online_tuner.record_performance(rspd_now, v_meas, e_k, u, t_deepc, exist_feasible_sol, cost)
                        else:
                            # Basic tuner
                            online_tuner.record_performance(e_k, u, t_deepc, exist_feasible_sol, cost)
                    
                    # Check if parameters should be updated
                    if online_tuner.should_update_parameters():
                        print(f"[OnlineTuning] Updating parameters at cycle {loop_count}")
                        
                        # Get new parameters (different interface for adaptive vs basic tuner)
                        if isinstance(online_tuner, AdaptiveOnlineTuner):
                            new_params = online_tuner.get_current_parameters()
                            metrics = {}  # Adaptive tuner doesn't return metrics directly
                        else:
                            new_params, metrics = online_tuner.update_parameters()
                        
                        # Update only the tunable parameters (keep structure parameters fixed)
                        Q_val = new_params.Q_val
                        R_val = new_params.R_val  
                        lambda_g_val = new_params.lambda_g_val
                        lambda_y_val = new_params.lambda_y_val
                        lambda_u_val = new_params.lambda_u_val
                        decay_rate_q = new_params.decay_rate_q
                        decay_rate_r = new_params.decay_rate_r
                        
                        # Recompute weighting matrices with new parameters
                        q_weights = Q_val * np.exp(-decay_rate_q * np.arange(THorizon))
                        r_weights = R_val * np.exp(-decay_rate_r * np.arange(THorizon))
                        Q = np.diag(q_weights)
                        R = np.diag(r_weights)
                        lambda_g = np.diag(np.tile(lambda_g_val, g_dim))
                        lambda_y = np.diag(np.tile(lambda_y_val, Tini))
                        lambda_u = np.diag(np.tile(lambda_u_val, Tini))
                        
                        last_param_update_cycle = loop_count
                        print(f"[OnlineTuning] Updated: Q={Q_val:.1f}, R={R_val:.3f}, λg={lambda_g_val:.1f}")
                
                # Main status printout
                controller_status = "DeePC" if not PID_control_activate else "PID"
                tuning_status = ""
                if online_tuner is not None and tuning_started:
                    tuning_status = f", Tuning=Active"
                
                print(
                    f"[{elapsed_time:.3f}s] "
                    f"v_ref={rspd_now:6.2f}, v_meas={v_meas:6.2f}, e={e_k:+6.2f}kph, "
                    f"u={u:6.2f}%, Controller={controller_status}{tuning_status}, "
                    f"solve_t={t_deepc:5.1f}ms, loop_t={actual_elapsed_time:5.1f}ms, "
                    f"freq={actual_control_frequency:5.1f}Hz, "
                    f"SOC={BMS_socMin:5.2f}%, cost={cost:.1f}"
                )

                # Debug purpose
                # print(
                #     "vref:", ref_horizon_speed,
                #     "u_init:", u_init,
                #     "y_init:", y_init,
                #     "Up_cur:", Up_cur,
                #     "Uf_cur:", Uf_cur,
                #     "Yp_cur:", Yp_cur,
                #     "Yf_cur:", Yf_cur,
                #     "g_opt:", g_opt,
                #     "u_opt:", u_opt
                # )
                # print(f"loop_count: {loop_count}")

                # ── 10) Save state for next iteration ──────────────────────────────
                prev_error     = e_k
                prev_ref_speed = rspd_now
                u_prev         = u
                # record Tinit length of historical data for state estimation
                u_history.append(u)                         
                spd_history.append(v_meas)
                u_init = np.array(u_history).reshape(-1, 1)  # shape (Tini,1)
                y_init = np.array(spd_history).reshape(-1, 1)

                # ── 11) Schedule next tick at 100 Hz ───────────────────────────────
                next_time += Ts
                loop_count += 1
                hankel_idx += 1
                # Update hankel_idx: because this is not ROTS system, 
                # there's lags, it's not running exactly Ts per loop, 
                # make hankel index correspond to the first 4 digits of elapsed_time
                # s = f"{elapsed_time:.3f}"               # "20.799"
                # digits = s.replace(".", "")             # "20799"
                # hankel_idx = int(digits[:3])            # 2079
               
                # 12) Append this tick’s values to log_data
                log_data.append({
                    "time":                 elapsed_time,
                    "v_ref":                rspd_now,
                    "v_meas":               v_meas,
                    "u":                    u,
                    "error":                e_k,
                    "t_deepc(ms)":          t_deepc,
                    "DeePC_Cost" :          cost,
                    "BMS_socMin":           BMS_socMin,
                    "SOC_CycleStarting":    SOC_CycleStarting,
                    "exist_feasible_sol":   exist_feasible_sol,
                    "DeePC_control":        DeePC_control,
                    "PID_control_activate": PID_control_activate,
                    "actual_elapsed_time":  actual_elapsed_time,
                    "hankel_idx":           hankel_idx,
                    "vref":                 ref_horizon_speed,
                    "u_init":               u_init,
                    "y_init":               y_init,
                    "Up_cur":               Up_cur,
                    "Uf_cur":               Uf_cur,
                    "Yp_cur":               Yp_cur,
                    "Yf_cur":               Yf_cur,
                    "g_opt" :               g_opt,
                    "u_opt" :               u_opt,
                    "q_weights":            q_weights,
                    "r_weights":            r_weights,
                    "Q_val":                Q_val,
                    "R_val":                R_val,
                    "lambda_g_val":         lambda_g_val,
                    "lambda_y_val":         lambda_y_val,
                    "lambda_u_val":         lambda_u_val,
                    "online_tuning_active": tuning_started if online_tuner else False,
                })
                if BMS_socMin <= SOC_Stop:
                    break

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt detected. Exiting…")

        finally:
            for ch in range(16):
                set_duty_cycle(bus, channel=ch, percent=0.0)                              # Zero out all PWM channels before exiting
            print("[Main] pca board PWM signal cleaned up and set back to 0.")
            
            # ── Save online tuning results ──────────────────────────────────
            if online_tuner is not None and tuning_started:
                tuning_results_file = PROJECT_DIR / f"online_tuning_results_{cycle_key}_{int(time.time())}.json"
                
                # Save results (different methods for different tuner types)
                if isinstance(online_tuner, AdaptiveOnlineTuner):
                    online_tuner.save_adaptive_results(str(tuning_results_file))
                    report = online_tuner.generate_adaptive_report()
                else:
                    online_tuner.save_tuning_results(str(tuning_results_file))
                    report = online_tuner.generate_tuning_report()
                print("\n" + "="*60)
                print("ONLINE PARAMETER TUNING COMPLETED")
                print("="*60)
                print(report)
                print("="*60)
                
                # Update config file with best parameters found
                if isinstance(online_tuner, AdaptiveOnlineTuner):
                    best_params = online_tuner.global_best_params
                    best_score = online_tuner.global_best_score
                    print(f"\nAdaptive tuning results:")
                    print(f"  Conditions encountered: {len(online_tuner.condition_performance)}")
                    print(f"  Total parameter updates: {online_tuner.total_updates}")
                else:
                    best_params, best_score = online_tuner.get_current_best()
                
                print(f"\nBest parameters found:")
                print(f"  Q_val: {best_params.Q_val:.2f}")
                print(f"  R_val: {best_params.R_val:.4f}")
                print(f"  lambda_g_val: {best_params.lambda_g_val:.2f}")
                print(f"  lambda_y_val: {best_params.lambda_y_val:.2f}")
                print(f"  lambda_u_val: {best_params.lambda_u_val:.2f}")
                print(f"  decay_rate_q: {best_params.decay_rate_q:.3f}")
                print(f"  decay_rate_r: {best_params.decay_rate_r:.3f}")
                print(f"  Best score achieved: {best_score:.3f}")

                # ── Save log_data to Excel ───────────────────────────────────
            if log_data:
                df = pd.DataFrame(log_data)
                df['cycle_name']   = cycle_key
                datetime = datetime.now()
                df['run_datetime'] = datetime.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_str = datetime.strftime("%H%M_%m%d")
                excel_filename = f"{timestamp_str}_DR_log_{veh_modelName}_{cycle_key}_Start{SOC_CycleStarting}%_{algorithm_name}_Ts{Ts}_Q{Q_val}_R{R_val}_decayQ{decay_rate_q}_decayR{decay_rate_r}_Tini{Tini}_gDim{g_dim}_λg{lambda_g_val}_λu{lambda_u_val}__λy{lambda_y_val}.xlsx"
                log_dir = os.path.join(base_folder, "Log_DriveRobot")
                os.makedirs(log_dir, exist_ok=True)     
                excel_path = os.path.join(log_dir, excel_filename)
                df.to_excel(excel_path, index=False)
                print(f"[Main] Saved log to '{excel_path}' as {excel_filename}")
        next_cycle = cycle_keys[idx+1] if idx+1 < len(cycle_keys) else None
        remaining_cycle = cycle_keys[idx+1:]
        print(f"[Main] Finish Running {cycle_key} on {veh_modelName}, Next running cycle {next_cycle}, take a 5 second break...")
        print(f"Current SOC: {BMS_socMin}%, system will stop at SOC: {SOC_Stop}% ")
        print(f"[Main] Plan to run the following cycles: {remaining_cycle}")
        time.sleep(5)

    # Stop CAN thread and wait up to 1 s
    dyno_can_running = False
    veh_can_running = False
    print("All CAN_Running Stops!!!")
    dyno_can_thread.join(timeout=1.0)
    veh_can_thread.join(timeout=1.0)
    bus.close()
    print("[Main] pca board PWM signal cleaned up and exited.")
    print("[Main] Cleaned up and exited.")
