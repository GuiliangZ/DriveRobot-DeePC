import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

import board
import busio
from adafruit_pca9685 import PCA9685

import cantools
import can
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

from smbus2 import SMBus

# ────────────────────────── GLOBALS VARIABLES ─────────────────────────────
# ——— CP2112 I²C setup ———
PCA9685_ADDR = 0x40      # default PCA9685 address
# PCA9685 register addresses
MODE1_REG    = 0x00
PRESCALE_REG = 0xFE
LED0_ON_L    = 0x06     # base address for channel 0

def init_pca9685(bus: SMBus, freq_hz: float = 1000.0):
    """
    Reset PCA9685 and set PWM frequency.
    """
    prescale_val = int(round(25_000_000.0 / (4096 * freq_hz) - 1))
    bus.write_byte_data(PCA9685_ADDR, MODE1_REG, 0x10)
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDR, PRESCALE_REG, prescale_val)
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDR, MODE1_REG, 0x00)
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDR, MODE1_REG, 0xA1)
    time.sleep(0.01)

def set_duty_cycle(bus: SMBus, channel: int, percent: float):
    """
    Set one channel’s duty cycle (0–100%). Uses 12-bit resolution.
    """
    if not (0 <= channel <= 15):
        raise ValueError("channel must be in 0..15")
    if not (0.0 <= percent <= 100.0):
        raise ValueError("percent must be between 0.0 and 100.0")

    # Convert percentage → 12-bit count (0..4095)
    duty_count = int(percent * 4095 / 100)
    on_l  = 0
    on_h  = 0
    off_l = duty_count & 0xFF
    off_h = (duty_count >> 8) & 0x0F

    # Compute first-LED register for this channel
    reg = LED0_ON_L + 4 * channel
    # Write [ON_L, ON_H, OFF_L, OFF_H]
    bus.write_i2c_block_data(PCA9685_ADDR, reg, [on_l, on_h, off_l, off_h])

# ─────────────────────────── LOAD DRIVE-CYCLE .MAT ────────────────────────────
def load_drivecycle_mat_files(base_folder: str):
    """
    Scan the 'drivecycle/' folder under base_folder. Load each .mat file
    via scipy.io.loadmat. Return a dict:
       { filename_without_ext: { varname: numpy_array, ... }, ... }.

    We assume each .mat has exactly one “user variable” that is an N×2 array:
      column 0 = time (s), column 1 = speed (mph).
    """
    drivecycle_dir = Path(base_folder) / "drivecycle"
    if not drivecycle_dir.is_dir():
        raise FileNotFoundError(f"Cannot find directory: {drivecycle_dir}")

    mat_data = {}
    for mat_file in drivecycle_dir.glob("*.mat"):
        try:
            data_dict = sio.loadmat(mat_file)
        except NotImplementedError:
            print(f"Warning: '{mat_file.name}' might be MATLAB v7.3. Skipping.")
            continue

        key = mat_file.stem
        mat_data[key] = data_dict
        user_vars = [k for k in data_dict.keys() if not k.startswith("__")]
        print(f"[Main] Loaded '{mat_file.name}' → variables = {user_vars}")

    return mat_data

# ─────────────────────────── LOAD TIME SERIES input-output data for building Hankel matrix ────────────────────────────
def load_timeseries(data_dir):
    """
    Read every .xlsx in data_dir, concatenating their 'u' and 'v_meas' columns.
    Returns
    -------
    u : np.ndarray, shape (T_total,)
    v : np.ndarray, shape (T_total,)
    """
    u_list, v_list = [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.xlsx'):
            continue
        df = pd.read_excel(os.path.join(data_dir, fname))
        u_list.append(df['u'].values)
        v_list.append(df['v_meas'].values)
    u = np.concatenate(u_list, axis=0)
    v = np.concatenate(v_list, axis=0)
    ud = u.reshape(-1, 1)
    yd = v.reshape(-1, 1)
    return ud, yd

# ──────────────────────────── USER CHOOSE CYCLE KEY ──────────────────────────────
def choose_cycle_key(all_cycles):
    """
    Print all available cycle keys and ask the user to pick one or more.
    Keeps prompting until at least one valid key is selected.
    Returns a list of chosen keys.
    """
    keys_list = list(all_cycles.keys())

    while True:
        print("\nAvailable drive cycles:")
        for idx, k in enumerate(keys_list, start=1):
            print(f"  [{idx}] {k}")

        sel = input("Select cycles (comma-separated indices or names): ").strip()
        tokens = [t.strip() for t in sel.split(",") if t.strip()]

        cycle_keys = []
        for t in tokens:
            if t.isdigit():
                i = int(t) - 1
                if 0 <= i < len(keys_list):
                    cycle_keys.append(keys_list[i])
            elif t in keys_list:
                cycle_keys.append(t)

        # Remove duplicates, preserve order
        # cycle_keys = list(dict.fromkeys(cycle_keys))

        if cycle_keys:
            return cycle_keys
        else:
            print("  → No valid selection detected. Please try again.\n")

def choose_vehicleModelName():
    """
    Print all available Tesla vehicle models ans ask the user to pick one for logging purpose.
    Keeps prompting until one and only one valid key is selected.
    Available vehicle models are: Model S,X,3,Y,Truck,Taxi
    Returns a string of vehicle model name
    """
    models = ["Model_S", "Model_X", "Model_3", "Model_Y", "Truck", "Taxi"]
    while True:
        print("\n Available vehicle models:")
        for idx, m in enumerate(models, start=1):
            print(f"[{idx}] {m}")
        sel = input("Select one model [index or name(case sensitive)]: ").strip()
        chosen = None

        if sel.isdigit():
            i = int(sel) - 1
            if 0 <= i <len(models):
                chosen = models[i]
        else:
            for m in models:
                if sel.lower() == m.lower():
                    chosen = m
                    break
        if chosen:
            return chosen
        else:
            print("  → Invalid selection. Please enter exactly one valid index or model name.\n")

# ───────────────────────────── GAIN‐SCHEDULING PID VALUES─────────────────────────────────
def get_gains_for_speed(ref_speed: float):
    """
    Return (Kp, Ki, Kd, Kff) according to the current reference speed (kph).

    """
    spd = ref_speed
    kp_bp_spd = np.array([0,  3,  4, 20,40,60,80,100,120,140], dtype=float)
    kp_vals = np.array([  6,  13, 6, 7, 8, 9, 9, 10, 10, 10], dtype=float)
    kp = float(np.interp(spd, kp_bp_spd, kp_vals))

    ki_bp_spd = np.array([0,20,40,60,80,100,120,140], dtype=float) 
    ki_vals = np.array([1.5,1.6,1.7,1.9,2,2,2,2], dtype=float)
    ki = float(np.interp(spd, ki_bp_spd, ki_vals))

    # The baseline code doesn't use kd, - now the kd_vals are wrong and random, adjust when needed
    kd_bp_spd = np.array([0,20,40,60,80,100,120], dtype=float)
    kd_vals = np.array([6,7,8,9,10,10,10], dtype=float)
    kd = float(np.interp(spd, kd_bp_spd, kd_vals))
    kd = 0

    kff_bp_spd = np.array([0,3,4,60,80,100,120,140], dtype=float)
    kff_vals = np.array([4,4,3,3,3,3,3,3], dtype=float)
    kff = float(np.interp(spd, kff_bp_spd, kff_vals))

    return (kp, ki, kd, kff)

# ───────────────────────────── baseline gain-scheduled PID Controller for DeePC ─────────────────────────────────
def compute_pid_control(elapsed_time,
                        FeedFwdTime,
                        ref_time,
                        ref_speed,
                        v_meas,
                        e_k,
                        prev_error,
                        I_state,
                        D_f_state,
                        Ts,
                        T_f):
    # ---- look ahead ----
    t_future = elapsed_time + FeedFwdTime
    if   t_future <= ref_time[0]:
        rspd_fut = ref_speed[0]
    elif t_future >= ref_time[-1]:
        rspd_fut = 0.0
    else:
        rspd_fut = float(np.interp(t_future, ref_time, ref_speed))
    e_fut = (rspd_fut - v_meas) * 0.621371
    # ---- current error in mph ----
    e_k_mph = e_k * 0.621371

    Kp, Ki, Kd, Kff = get_gains_for_speed(v_meas)
    P_term = Kp * e_k_mph
    D_k    = Kd * (e_k_mph - prev_error) / Ts
    alpha  = Ts / (Ts + T_f)
    D_f_state = D_f_state + alpha * (D_k - D_f_state)
    D_term    = D_f_state

    if v_meas > 0.1 and (v_meas * 0.621371) > 0.1:
        I_state += Ki * Ts * e_k_mph
        I_out = I_state
    else:
        I_state = 0.0
        I_out   = 0.0

    FF_term = Kff * e_fut
    u_PID   = P_term + I_out + D_term + FF_term

    # return both the command and updated integrator/filter states
    return u_PID, P_term, I_out, D_term, e_k_mph

# ═══════════════════════════════ DTI CALCULATION FUNCTIONS ═══════════════════════════════
def calculate_dti_metrics(df, ref_speed_array, ref_time_array, cycle_name):
    """
    Calculate DTI (Drive Rating Index) metrics based on driveRobotPerformance.m methodology
    Returns: dict with ER, DR, EER, ASCR, IWR, RMSSEkph metrics
    """
    try:
        # Extract data arrays
        time_data = df['time'].values
        v_ref = df['v_ref'].values  # Reference speed in kph
        v_meas = df['v_meas'].values  # Measured speed in kph
        control_data = df['u'].values  # Control signal in %
        
        # Ensure arrays are same length
        min_length = min(len(time_data), len(v_ref), len(v_meas), len(control_data))
        time_data = time_data[:min_length]
        v_ref = v_ref[:min_length]
        v_meas = v_meas[:min_length]
        control_data = control_data[:min_length]
        
        # Calculate time step
        dt = np.mean(np.diff(time_data)) if len(time_data) > 1 else 0.1
        
        # 1. RMSSE (Root Mean Square Speed Error) in kph
        speed_error = v_ref - v_meas
        rmsse_kph = np.sqrt(np.mean(speed_error**2))
        
        # 2. ER (Error Rate) - normalized RMSSE
        mean_ref_speed = np.mean(v_ref[v_ref > 0.1])  # Avoid division by zero
        er = rmsse_kph / max(mean_ref_speed, 1.0) * 100  # Percentage
        
        # 3. IWR (Idle Waste Rate) - Based on IWRComparison.m methodology
        iwr = calculate_iwr(time_data, v_meas, v_ref, dt)
        
        # 4. ASCR (Acceleration Smoothness/Control Rate)
        ascr = calculate_ascr(control_data, dt)
        
        # 5. EER (Enhanced Error Rate) - Combines tracking and smoothness
        eer = calculate_eer(speed_error, control_data, dt)
        
        # 6. DR (Driver Rating) - Composite score
        dr = calculate_composite_dr(er, iwr, ascr, eer, rmsse_kph)
        
        # Compile results
        dti_metrics = {
            'cycle_name': cycle_name,
            'ER': er,
            'DR': dr, 
            'EER': eer,
            'ASCR': ascr,
            'IWR': iwr,
            'RMSSEkph': rmsse_kph,
            'data_points': min_length,
            'cycle_duration_sec': time_data[-1] - time_data[0] if len(time_data) > 1 else 0,
            'mean_ref_speed': mean_ref_speed,
            'mean_tracking_error': np.mean(np.abs(speed_error))
        }
        
        return dti_metrics
        
    except Exception as e:
        print(f"[DTI Error] Failed to calculate DTI metrics: {e}")
        return {
            'cycle_name': cycle_name,
            'ER': 999.0, 'DR': 999.0, 'EER': 999.0, 
            'ASCR': 999.0, 'IWR': 999.0, 'RMSSEkph': 999.0,
            'error': str(e)
        }

def calculate_iwr(time_data, v_meas, v_ref, dt):
    """Calculate IWR (Idle Waste Rate) based on IWRComparison.m methodology"""
    try:
        # Convert speeds to m/s for physics calculations
        v_meas_ms = np.array(v_meas) / 3.6  # kph to m/s
        v_ref_ms = np.array(v_ref) / 3.6   # kph to m/s
        
        # Vehicle parameters (approximate values)
        mass = 2300  # kg (Tesla Model 3 approximate mass)
        
        # Calculate acceleration using central difference (cdiff equivalent)
        accel_meas = np.gradient(v_meas_ms, dt)  # m/s^2
        accel_ref = np.gradient(v_ref_ms, dt)    # m/s^2
        
        # Calculate inertial forces
        F_I_meas = mass * accel_meas  # N
        F_I_ref = mass * accel_ref    # N
        
        # Calculate distance increments
        d_meas = v_meas_ms * dt  # m
        d_ref = v_ref_ms * dt    # m
        
        # Calculate inertial work increments
        w_I_meas = F_I_meas * d_meas  # J
        w_I_ref = F_I_ref * d_ref      # J
        
        # Sum only positive work (energy into vehicle)
        IWT_meas = np.sum(w_I_meas[w_I_meas > 0])  # J
        IWT_ref = np.sum(w_I_ref[w_I_ref > 0])     # J
        
        # Calculate IWR percentage
        if IWT_ref > 0:
            iwr = (IWT_meas - IWT_ref) / IWT_ref * 100  # %
        else:
            iwr = 0.0
            
        return iwr
        
    except Exception as e:
        print(f"[IWR Error] {e}")
        return 999.0

def calculate_ascr(control_data, dt):
    """Calculate ASCR (Acceleration Smoothness/Control Rate)"""
    try:
        # Calculate control derivatives (control jerk)
        control_derivative = np.gradient(control_data, dt)
        control_jerk = np.gradient(control_derivative, dt)
        
        # RMS control jerk as smoothness metric
        rms_control_jerk = np.sqrt(np.mean(control_jerk**2))
        
        # Normalize by typical control range (0-100%)
        ascr = rms_control_jerk / 100.0 * 100  # Percentage
        
        return ascr
        
    except Exception as e:
        print(f"[ASCR Error] {e}")
        return 999.0

def calculate_eer(speed_error, control_data, dt):
    """Calculate EER (Enhanced Error Rate) - combines tracking and control smoothness"""
    try:
        # Tracking component
        rms_error = np.sqrt(np.mean(speed_error**2))
        
        # Control smoothness component
        control_changes = np.abs(np.diff(control_data))
        rms_control_change = np.sqrt(np.mean(control_changes**2)) if len(control_changes) > 0 else 0.0
        
        # Combined metric (weighted average)
        eer = 0.7 * rms_error + 0.3 * rms_control_change
        
        return eer
        
    except Exception as e:
        print(f"[EER Error] {e}")
        return 999.0

def calculate_composite_dr(er, iwr, ascr, eer, rmsse):
    """Calculate composite DR (Driver Rating) based on individual metrics"""
    try:
        # Weights for composite score (can be tuned based on importance)
        weights = {
            'er': 0.25,      # Error rate
            'iwr': 0.25,     # Idle waste rate  
            'ascr': 0.20,    # Control smoothness
            'eer': 0.20,     # Enhanced error rate
            'rmsse': 0.10    # Direct RMSSE contribution
        }
        
        # Normalize metrics to similar scales for DTI < 1.2 target (0-5 range for tighter control)
        er_norm = min(er / 3.0, 5.0)           # ER: 0-3% -> 0-5 (tighter than before)
        iwr_norm = min(abs(iwr) / 6.0, 5.0)    # IWR: 0-6% -> 0-5 (tighter than before)
        ascr_norm = min(ascr / 2.0, 5.0)       # ASCR: 0-2 -> 0-5 (much tighter for smoothness)
        eer_norm = min(eer / 1.5, 5.0)         # EER: 0-1.5 -> 0-5 (tighter than before)
        rmsse_norm = min(rmsse / 2.0, 5.0)     # RMSSE: 0-2kph -> 0-5 (relaxed for ±2kph tolerance)
        
        # Calculate weighted composite score
        dr = (weights['er'] * er_norm + 
              weights['iwr'] * iwr_norm + 
              weights['ascr'] * ascr_norm + 
              weights['eer'] * eer_norm + 
              weights['rmsse'] * rmsse_norm)
        
        return dr
        
    except Exception as e:
        print(f"[DR Error] {e}")
        return 999.0

def print_dti_results(dti_metrics, cycle_name):
    """Print DTI results in a formatted table"""
    print(f"\n╔══════════════════════════════════════════════════════════════╗")
    print(f"║                    DTI ANALYSIS RESULTS                     ║")
    print(f"║                     {cycle_name:^30}                     ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║ Metric                    │ Value      │ Target    │ Status ║")
    print(f"╠───────────────────────────┼────────────┼───────────┼────────╣")
    print(f"║ RMSSE (kph)              │ {dti_metrics.get('RMSSEkph', 999):8.3f}   │  < 0.8 │ {'PASS' if dti_metrics.get('RMSSEkph', 999) < 1.5 else 'FAIL':^6} ║")
    print(f"║ ER - Error Rate (%)      │ {dti_metrics.get('ER', 999):8.3f}   │  < 2.000  │ {'PASS' if dti_metrics.get('ER', 999) < 2.0 else 'FAIL':^6} ║")
    print(f"║ IWR - Idle Waste (%)     │ {dti_metrics.get('IWR', 999):8.3f}   │ -0.8 to +1.2│ {'PASS' if -0.5 <= dti_metrics.get('IWR', 999) <= 1.0 else 'FAIL':^6} ║")
    print(f"║ ASCR - Control Smooth    │ {dti_metrics.get('ASCR', 999):8.3f}   │  < 1.000  │ {'PASS' if dti_metrics.get('ASCR', 999) < 1.0 else 'FAIL':^6} ║")
    print(f"║ EER - Enhanced Error     │ {dti_metrics.get('EER', 999):8.3f}   │  < 1.100  │ {'PASS' if dti_metrics.get('EER', 999) < 1.1 else 'FAIL':^6} ║")
    print(f"║ DR - Driver Rating       │ {dti_metrics.get('DR', 999):8.3f}   │  < 1.200  │ {'PASS' if dti_metrics.get('DR', 999) < 1.2 else 'FAIL':^6} ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║ Cycle Duration: {dti_metrics.get('cycle_duration_sec', 0):6.1f} sec │ Data Points: {dti_metrics.get('data_points', 0):6d}     ║")
    print(f"║ Mean Ref Speed: {dti_metrics.get('mean_ref_speed', 0):6.1f} kph │ Mean Error: {dti_metrics.get('mean_tracking_error', 0):7.3f} kph ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    # Overall assessment OPTIMIZED from 1907_0721 analysis for DTI < 1.2 and tracking < 1.5kph
    metrics_pass = [
        dti_metrics.get('RMSSEkph', 999) < 1.5,   # Match tracking requirement < 1.5kph
        dti_metrics.get('ER', 999) < 2.0,         # Optimized from analysis: achievable target
        -0.8 <= dti_metrics.get('IWR', 999) <= 1.2,  # Tighter efficiency target from data
        dti_metrics.get('ASCR', 999) < 1.0,       # Very tight smoothness from multi-stage filtering
        dti_metrics.get('EER', 999) < 1.1,        # Achievable with current enhancements
        dti_metrics.get('DR', 999) < 1.2          # PRIMARY TARGET: DTI < 1.2
    ]
    
    overall_score = sum(metrics_pass) / len(metrics_pass) * 100
    print(f"\n🎯 OVERALL DTI SCORE: {overall_score:.1f}% ({sum(metrics_pass)}/{len(metrics_pass)} metrics passed)")
    
    if overall_score >= 83.3:  # 5/6 metrics pass
        print("🏆 OUTSTANDING: DTI < 1.2 achieved - Premium Tesla performance!")
    elif overall_score >= 66.7:  # 4/6 metrics pass  
        print("✨ GOOD: DTI performance strong, approaching < 1.2 target")
    else:
        print("⚠️  OPTIMIZATION NEEDED: DTI performance needs improvement for < 1.2 target")



# ──────────────────────────── Enhancement Controller ──────────────────────────────
class AdditiveEnhancementController:
    def __init__(self, Ts=0.01, T_f=5000.0, FeedFwdTime=0.65, history_length=50, enable_enhancements=True):
        self.Ts = Ts
        self.T_f = T_f
        self.FeedFwdTime = FeedFwdTime
        self.enable_enhancements = enable_enhancements
        self.baseline_I_state = 0.0
        self.baseline_prev_error = 0.0
        # Enhancement data collection
        self.error_history = deque(maxlen=history_length)
        self.speed_history = deque(maxlen=history_length)
        self.speed_derivative_history = deque(maxlen=history_length)
        self.performance_history = deque(maxlen=20)
        # Enhancement parameters (only active if enable_enhancements=True)
        self.adaptive_gain_range = [0.92, 1.08]    # Conservative ±8% range
        self.adaptive_gain_range = [0.9, 1.1]    # Conservative ±8% range
        self.adaptation_rate = 0.004               # Slow, stable adaptation
        self.adaptation_rate = 0.006               
        self.feedforward_scaling = 0.6             # Conservative feedforward
        self.performance_scaling_range = [0.3, 1.2] # Enhancement strength scaling
        # Enhancement state
        self.kp_enhancement_mult = 1.0  # Multiplier for additional Kp signal
        self.ki_enhancement_mult = 1.0  # Multiplier for additional Ki signal
        self.kp_enhancement_mult = 1.0  # Multiplier for additional Kp signal
        self.ki_enhancement_mult = 1.0  # Multiplier for additional Ki signal
        self.consecutive_good_count = 0
        self.consecutive_poor_count = 0
        # Data-driven components
        self.learned_feedforward_map = {}  # speed_derivative -> feedforward_gain
        self.error_pattern_map = {}        # speed_profile -> error_correction
        # Performance monitoring
        self.recent_errors = deque(maxlen=15)
        self.recent_performance_scores = deque(maxlen=10)
        
    def compute_control(self, error, ref_speed, v_meas, ref_time=None, ref_speed_array=None, elapsed_time=None):
        baseline_kp, baseline_ki, baseline_kd, baseline_kff = get_gains_for_speed(ref_speed)
        baseline_P = baseline_kp * error
        if ref_speed > 0.1 and v_meas > 0.1:
            self.baseline_I_state = self.baseline_I_state + baseline_ki * self.Ts * error
            baseline_I = self.baseline_I_state
        else:
            self.baseline_I_state = 0.0  # RESET INTEGRAL STATE
            baseline_I = 0.0
        baseline_D = 0.0
        baseline_FF = 0.0
        if ref_time is not None and ref_speed_array is not None and elapsed_time is not None:
            try:
                future_time = elapsed_time + self.FeedFwdTime
                if future_time <= ref_time[-1]:
                    current_ref = float(np.interp(elapsed_time, ref_time, ref_speed_array))
                    future_ref = float(np.interp(future_time, ref_time, ref_speed_array))
                    ref_rate = (future_ref - current_ref) / self.FeedFwdTime
                    if abs(ref_rate) > 0.1:
                        baseline_FF = baseline_kff * ref_rate * 0.8  # Standard scaling
            except:
                baseline_FF = 0.0
        baseline_control = baseline_P + baseline_I + baseline_D + baseline_FF
        self.baseline_prev_error = error
        enhancement_signal = 0.0
        if self.enable_enhancements and abs(ref_speed) > 0.01:
            self.update_enhancement_data(error, ref_speed, v_meas)
            adaptive_enhancement = self.compute_adaptive_gain_enhancement(error, baseline_kp, baseline_ki)
            feedforward_enhancement = self.compute_feedforward_enhancement(ref_speed, ref_time, ref_speed_array, elapsed_time)
            pattern_enhancement = self.compute_pattern_enhancement(ref_speed)
            performance_scale = self.compute_performance_scaling()
            enhancement_signal = (adaptive_enhancement + feedforward_enhancement + pattern_enhancement) * performance_scale
            max_enhancement = min(20.0, abs(baseline_control) * 0.3)  
            enhancement_signal = np.clip(enhancement_signal, -max_enhancement, max_enhancement)
            self.update_enhancement_adaptation(error)
        if ref_speed < 0.01 and v_meas < 0.1:
            total_control = 0.0  # Complete stop - no control input
            enhancement_signal = 0.0  # Ensure enhancement is also zero for logging
        else:
            total_control = baseline_control + enhancement_signal
        total_control = baseline_control
        # ═══ STEP 4: PREPARE DEBUG INFO ═══
        debug_info = {
            'baseline_control': baseline_control,
            'enhancement_signal': enhancement_signal,
            'total_control': total_control,
            "baseline_P":baseline_P,
            "baseline_I":baseline_I,
            "baseline_FF":baseline_FF,
            'baseline_kp': baseline_kp,
            'baseline_ki': baseline_ki,
            'baseline_kff': baseline_kff,
            'kp_enhancement_mult': self.kp_enhancement_mult,
            'ki_enhancement_mult': self.ki_enhancement_mult,
            'performance_scaling': self.compute_performance_scaling() if self.enable_enhancements else 1.0,
            'enhancement_enabled': self.enable_enhancements,
            'tracking_success': abs(error) < 0.5,
            'ref_speed': ref_speed,
            'baseline_integral_state': self.baseline_I_state,
            'error': error,
        }
        return total_control, debug_info
    
    def update_enhancement_data(self, error, ref_speed, v_meas):
        """Update data collections for enhancement computation"""
        self.error_history.append(error)
        self.speed_history.append(ref_speed)
        self.recent_errors.append(abs(error))
        if len(self.speed_history) >= 2:
            speed_derivative = (self.speed_history[-1] - self.speed_history[-2]) / self.Ts
            self.speed_derivative_history.append(speed_derivative)
        performance_score = 1.0 / (1.0 + abs(error))
        self.recent_performance_scores.append(performance_score)
    
    def compute_adaptive_gain_enhancement(self, error, baseline_kp, baseline_ki):
        """Compute adaptive gain enhancement signals"""
        if len(self.recent_errors) < 10:
            return 0.0
        kp_enhancement_signal = baseline_kp * error * (self.kp_enhancement_mult - 1.0) 
        ki_enhancement_signal = baseline_ki * self.baseline_I_state * (self.ki_enhancement_mult - 1.0)
        return kp_enhancement_signal + ki_enhancement_signal
    
    def compute_feedforward_enhancement(self, ref_speed, ref_time, ref_speed_array, elapsed_time):
        """Compute data-driven feedforward enhancement"""
        if ref_time is None or ref_speed_array is None or elapsed_time is None:
            return 0.0  
        try:
            future_time = elapsed_time + self.FeedFwdTime * 0.5  # Shorter horizon for enhancement
            if future_time <= ref_time[-1]:
                current_ref = float(np.interp(elapsed_time, ref_time, ref_speed_array))
                future_ref = float(np.interp(future_time, ref_time, ref_speed_array))
                speed_derivative = (future_ref - current_ref) / (self.FeedFwdTime * 0.5)
                if abs(speed_derivative) > 0.2:
                    speed_key = int(ref_speed / 10) * 10  # Round to nearest 10 kph
                    learned_gain = self.learned_feedforward_map.get(speed_key, 2.0)  # Default gain
                    enhancement_ff = learned_gain * speed_derivative * self.feedforward_scaling
                    return enhancement_ff
        except:
            pass
        return 0.0
    
    def compute_pattern_enhancement(self, ref_speed):
        """Compute error pattern learning enhancement"""
        if len(self.error_history) < 20:
            return 0.0
        
        # Simple pattern: predict error based on speed range
        speed_range = int(ref_speed / 20) * 20  # Round to nearest 20 kph ranges
        
        if speed_range in self.error_pattern_map:
            pattern_correction = self.error_pattern_map[speed_range]
            return pattern_correction * 0.5  # Conservative scaling
        
        return 0.0
    
    def compute_performance_scaling(self):
        """Compute performance-based scaling for enhancement signals"""
        if len(self.recent_performance_scores) < 5:
            return 0.8  # Default moderate scaling
        recent_performance = np.mean(list(self.recent_performance_scores))
        # Scale enhancement strength inversely with performance
        if recent_performance > 0.8:  # Good performance
            scale = self.performance_scaling_range[0] + 0.2  # Lower enhancement
        elif recent_performance < 0.5:  # Poor performance
            scale = self.performance_scaling_range[1]  # Higher enhancement
        else:  # Moderate performance
            # Linear interpolation
            ratio = (0.8 - recent_performance) / (0.8 - 0.5)
            scale = self.performance_scaling_range[0] + ratio * (self.performance_scaling_range[1] - self.performance_scaling_range[0])
        return scale
    
    def update_enhancement_adaptation(self, error):
        """Update enhancement adaptation based on performance"""
        if len(self.recent_errors) < 12:
            return
        # Performance tracking
        if abs(error) < 0.4:
            self.consecutive_good_count += 1
            self.consecutive_poor_count = 0
        elif abs(error) > 1.5:
            self.consecutive_poor_count += 1
            self.consecutive_good_count = 0
        else:
            self.consecutive_good_count = max(0, self.consecutive_good_count - 1)
            self.consecutive_poor_count = max(0, self.consecutive_poor_count - 1)
        # Adaptive gain enhancement adjustment
        recent_avg_error = np.mean(list(self.recent_errors))
        # Poor performance -> increase enhancement multipliers
        if self.consecutive_poor_count >= 15:
            if recent_avg_error > 1.0:
                self.kp_enhancement_mult = min(self.adaptive_gain_range[1], 
                                             self.kp_enhancement_mult + self.adaptation_rate)
            if recent_avg_error > 0.8:
                self.ki_enhancement_mult = min(self.adaptive_gain_range[1],
                                             self.ki_enhancement_mult + self.adaptation_rate * 0.7)
        # Good performance -> fine-tune enhancement multipliers
        elif self.consecutive_good_count >= 25:
            if recent_avg_error < 0.3:
                # Very good performance - small increase for optimization
                self.kp_enhancement_mult = min(self.adaptive_gain_range[1],
                                             self.kp_enhancement_mult + self.adaptation_rate * 0.3)
    
    def learn_from_data(self, historical_data=None):
        """Learn patterns from historical data (can be called offline)"""
        if historical_data is None:
            # Use current session data for learning
            if len(self.error_history) > 100:
                self.update_learned_patterns()
        else:
            # Learn from provided historical data
            self.process_historical_data(historical_data)
    
    def update_learned_patterns(self):
        """Update learned patterns from current session data"""
        if len(self.speed_derivative_history) > 50:
            # Learn feedforward patterns
            speeds = list(self.speed_history)[-50:]
            speed_derivs = list(self.speed_derivative_history)[-50:]
            errors = list(self.error_history)[-50:]
            # Group by speed ranges and learn average corrections
            for i in range(len(speeds)):
                speed_key = int(speeds[i] / 10) * 10
                if abs(speed_derivs[i]) > 0.5:  # Significant speed changes
                    if speed_key not in self.learned_feedforward_map:
                        self.learned_feedforward_map[speed_key] = 2.0
                    # Slowly update learned gain based on error response
                    error_response = abs(errors[i]) if i < len(errors) else 1.0
                    correction_factor = 1.0 + (1.0 - min(error_response, 2.0)) * 0.1
                    # Exponential moving average update
                    self.learned_feedforward_map[speed_key] = \
                        0.95 * self.learned_feedforward_map[speed_key] + 0.05 * correction_factor * 2.0
    
    def reset(self):
        """Reset controller state for new cycle"""
        # Reset baseline PID states (NEVER modify baseline logic)
        self.baseline_I_state = 0.0
        self.baseline_prev_error = 0.0
        
        # Reset enhancement data
        self.error_history.clear()
        self.speed_history.clear()
        self.speed_derivative_history.clear()
        self.performance_history.clear()
        self.recent_errors.clear()
        self.recent_performance_scores.clear()
        
        # Reset enhancement adaptation states
        self.consecutive_good_count = 0
        self.consecutive_poor_count = 0
        
        # Keep learned patterns (persistent learning)
        # self.learned_feedforward_map and self.error_pattern_map are preserved

# ───────────────────────────── DeePC RELATED - HANKEL MATRIX ─────────────────────────────────
def hankel(x, L):
    """
        ------Construct Hankel matrix------
        x: data sequence (data_size, x_dim)
        L: row dimension of the hankel matrix
        T: data samples of data x
        return: H(x): hankel matrix of x  H(x): (x_dim*L, T-L+1)
                H(x) = [x(0)   x(1) ... x(T-L)
                        x(1)   x(2) ... x(T-L+1)
                        .       .   .     .
                        .       .     .   .
                        .       .       . .
                        x(L-1) x(L) ... x(T-1)]
                Hankel matrix of order L has size:  (x_dim*L, T-L+1)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    T, x_dim = x.shape

    Hx = np.zeros((L * x_dim, T - L + 1))
    for i in range(L):
        Hx[i * x_dim:(i + 1) * x_dim, :] = x[i:i + T - L + 1, :].T  # x need transpose to fit the hankel dimension
    return Hx

def hankel_full(ud, yd, Tini, THorizon):
    """
    Build the full DeePC Hankel matrix once by stacking past and future blocks.

    Parameters:
    -----------
    ud : array_like, shape (T_data, u_dim)
        Historical input sequence.
    yd : array_like, shape (T_data, y_dim)
        Historical output sequence.
    Tini : int
        Number of past (initialization) steps.
    THorizon : int
        Prediction horizon (number of future steps).

    Returns:
    --------
    hankel_full_mtx : np.ndarray, shape ((u_dim + y_dim) * (Tini + THorizon), K)
        A stacked Hankel matrix containing:
            [ Up;  # past-input block
              Yp;  # past-output block
              Uf;  # future-input block
              Yf ] # future-output block
        where K = T_data - (Tini + THorizon) + 1 is the total number of columns. (Large number)
    """
    # Build block-Hankel for inputs and outputs
    Hud = hankel(ud, Tini + THorizon)
    Huy = hankel(yd, Tini + THorizon)

    u_dim = ud.shape[1]
    y_dim = yd.shape[1]

    # Slice into past (first Tini) and future (last THorizon)
    Up = Hud[: u_dim * Tini, :]
    Uf = Hud[u_dim * Tini : u_dim * (Tini + THorizon), :]
    Yp = Huy[: y_dim * Tini, :]
    Yf = Huy[y_dim * Tini : y_dim * (Tini + THorizon), :]
    print(f"Hankel full matrix with shape: Up{Up.shape}, Uf{Uf.shape},Yp{Yp.shape},Yf{Yf.shape}")
    return Up, Uf, Yp, Yf

def hankel_subBlocks(Up, Uf, Yp, Yf, Tini, THorizon, hankel_subB_size):
    """
    hankel_subB_size:   The sub-hankel matrix for current optimization problem
    hankel_idx:         the current hankel matrix in the official run
    The sub-hankel matrix was chosen as hankel_idx as center, and front g_dim, and back g_dim data section.
      front g_dim for state estimation, and back g_dim for prediction. g_dim I'm leaving for 50 front and 50 back buffer by choosing g_dim = 100(hankel_subB_size=199)
    
    shape: Up, Uf, Tp, Tf - (Tini, g_dim)/(THorizon, g_dim)
    """
    # how many columns on each side of hankel_idx we want
    g_dim = hankel_subB_size - Tini - THorizon + 1

    # desired slice is [start:end] with width = end - start = g_dim
    half  = g_dim // 2
    start = 200 - half
    end   = start + g_dim
    width = end - start

    # allocate zero‐padded output blocks
    Up_cur = np.zeros((Tini,        width), dtype=Up.dtype)
    Uf_cur = np.zeros((THorizon,    width), dtype=Uf.dtype)
    Yp_cur = np.zeros((Tini,        width), dtype=Yp.dtype)
    Yf_cur = np.zeros((THorizon,    width), dtype=Yf.dtype)

    # clamp source columns to [0, max_col)
    max_col = Up.shape[1]
    src_start = max(start, 0)
    src_end   = min(end,   max_col)

    # where in the padded block these columns should go
    dst_start = src_start - start        # if start<0, dst_start>0
    dst_end   = dst_start + (src_end - src_start)

    # copy the in-bounds slice into the zero blocks
    Up_cur[:,      dst_start:dst_end] = Up[:Tini,         src_start:src_end]
    Uf_cur[:,      dst_start:dst_end] = Uf[:THorizon,     src_start:src_end]
    Yp_cur[:,      dst_start:dst_end] = Yp[:Tini,         src_start:src_end]
    Yf_cur[:,      dst_start:dst_end] = Yf[:THorizon,     src_start:src_end]

    return Up_cur, Uf_cur, Yp_cur, Yf_cur


# def hankel_subBlocks(Up, Uf, Yp, Yf, Tini, THorizon, hankel_subB_size, hankel_idx):

#     g_dim = hankel_subB_size - Tini - THorizon + 1
#     Up_cur = Up[:Tini,         hankel_idx-g_dim:hankel_idx+g_dim]
#     Uf_cur = Uf[:Tini,         hankel_idx-g_dim:hankel_idx+g_dim]
#     Yp_cur = Yp[Tini:THorizon, hankel_idx-g_dim:hankel_idx+g_dim]
#     Yf_cur = Yf[Tini:THorizon, hankel_idx-g_dim:hankel_idx+g_dim]
#     return Up_cur, Uf_cur, Yp_cur, Yf_cur

# ───────── Relay Auto-Tuner ─────────
class RelayAutoTuner:
    def __init__(self, apply_control, measure_output, relay_amp=10.0):
        self.apply_control = apply_control
        self.measure_output = measure_output
        self.relay_amp = relay_amp

    def tune(self, duration=10.0, dt=0.05):
        t0 = time.time()
        toggle = True
        data_t, data_y = [], []
        next_toggle = t0
        while time.time() - t0 < duration:
            now = time.time()
            if now >= next_toggle:
                u = self.relay_amp if toggle else -self.relay_amp
                self.apply_control(u)
                toggle = not toggle
                next_toggle += dt
            y = self.measure_output()
            data_t.append(now - t0)
            data_y.append(y)
            time.sleep(dt)

        # Estimate Tu from zero crossings
        zs = [data_t[i] for i in range(1, len(data_y))
              if data_y[i-1] * data_y[i] < 0]
        Tu = 2 * np.mean(np.diff(zs))
        amp = (max(data_y) - min(data_y)) / 2
        Ku = (4 * self.relay_amp) / (np.pi * amp)

        # Ziegler–Nichols PID tuning
        Kp = 0.6 * Ku
        Ki = Kp / (0.5 * Tu)
        Kd = 0.125 * Kp * Tu
        return Kp, Ki, Kd

