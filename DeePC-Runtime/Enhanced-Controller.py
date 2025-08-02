#!/usr/bin/env python3
"""
Name: Enhanced-Controller.py
Author: Guiliang Zheng
Date:at 22/07/2025
version: 1.0.0
Description: Additive Enhancement Controller

Enhancement Components:
1. Adaptive Gain Enhancement (based on performance)
2. Data-Driven Feedforward (from historical patterns)
3. Error Pattern Learning (predictive corrections)
4. Performance-Based Scaling (dynamic enhancement strength)

Note:
The calculate WTI displayed at the end of drive cycle
 are typically higher than official dyno data processed by matlab.
EX. IWR typically 0.5-1.0 higher than actual IWR

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

import cantools
import can
from utils_home import *

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
algorithm_name = "Enhanced-Controller-New"
latest_speed = None
latest_force = None
dyno_can_running = True
veh_can_running = True
BMS_socMin = None
CP2112_BUS = 3

# ──────────────────────────── CAN LISTENER THREADS ──────────────────────────────
def dyno_can_listener_thread(dbc_path: str, can_iface: str):
    """Dyno CAN listener - identical to proven baseline"""
    global latest_speed, latest_force, dyno_can_running

    try:
        db = cantools.database.load_file(dbc_path)
        speed_force_msg = db.get_message_by_name('Speed_and_Force')
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
    except Exception as e:
        print("[CAN Thread] ERROR: {}".format(e))
        return

    while dyno_can_running:
        msg = bus.recv(timeout=1.0)
        if msg is None or msg.arbitration_id != speed_force_msg.frame_id:
            continue
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
            s = decoded.get('Speed_kph')
            if s is not None:
                latest_speed = float(s) if abs(s) > 0.1 else 0.0
        except:
            continue
    bus.shutdown()

def veh_can_listener_thread(dbc_path: str, can_iface: str):
    """Vehicle CAN listener - identical to proven baseline"""
    global BMS_socMin, veh_can_running
    try:
        db = cantools.database.load_file(dbc_path)
        bms_soc_msg = db.get_message_by_name('BMS_socStatus')
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
    except Exception as e:
        print("[CAN Thread] ERROR: {}".format(e))
        return

    while veh_can_running:
        msg = bus.recv(timeout=3.0)
        if msg is None or msg.arbitration_id != bms_soc_msg.frame_id:
            continue
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
            BMS_socMin = decoded.get('BMS_socMin')
        except:
            continue
    bus.shutdown()


# ─────────────────────────────── MAIN CONTROL ─────────────────────────────────
if __name__ == "__main__":
    # System parameters (identical to proven baseline)
    max_delta = 80.0
    SOC_CycleStarting = 0.0
    SOC_Stop = 2.2
    Ts = 0.01
    T_f = 5000.0
    FeedFwdTime = 0.65
    
    # Initialize Additive Enhancement Controller
    additive_controller = AdditiveEnhancementController(
        Ts=Ts, 
        T_f=T_f, 
        FeedFwdTime=FeedFwdTime,
        enable_enhancements=True  
    )
    
    # ─── PCA9685 PWM SETUP ──────────────────────────────────────────────────────
    bus = SMBus(CP2112_BUS)
    init_pca9685(bus, freq_hz=1000.0)

    # ─── START CAN LISTENER THREADS ───────────────────────────────────────────────
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
    all_cycles = load_drivecycle_mat_files(base_folder)
    cycle_keys = choose_cycle_key(all_cycles)
    veh_modelName = choose_vehicleModelName()

    for idx, cycle_key in enumerate(cycle_keys):
        # SOC management
        if BMS_socMin is not None and BMS_socMin <= SOC_Stop:
            break
        else:
            SOC_CycleStarting = BMS_socMin if BMS_socMin is not None else 0.0
            SOC_CycleStarting = round(SOC_CycleStarting, 2)

        # Load cycle data
        cycle_data = all_cycles[cycle_key]
        print("\n[Main] Using reference cycle '{}'".format(cycle_key))
        mat_vars = [k for k in cycle_data.keys() if not k.startswith("__")]
        if len(mat_vars) != 1:
            raise RuntimeError(f"Expected exactly one variable in '{cycle_key}', found {mat_vars}")
        varname = mat_vars[0]
        ref_array = cycle_data[varname]
        if ref_array.ndim != 2 or ref_array.shape[1] < 2:
            raise RuntimeError(f"Expected '{varname}' to be N×2 array. Got shape {ref_array.shape}")

        # Extract reference data
        ref_time = ref_array[:, 0].astype(float).flatten()
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()
        ref_speed = ref_speed_mph * 1.60934  # Convert to kph
        print(f"[Main] Reference loaded: shape = {ref_array.shape}")

        # Reset for new cycle
        loop_count = 0
        additive_controller.reset()
        log_data = []
        next_time = time.perf_counter()
        t0 = time.perf_counter()
        print(f"\n[Main] Starting cycle '{cycle_key}' on {veh_modelName}, duration={ref_time[-1]:.2f}s")

        # Real-time scheduling
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
            print("[Main] Real-time scheduling enabled")
        except:
            print("Need root for real-time scheduling")

        # ─── MAIN 100 Hz CONTROL LOOP ─────────────────────────────────────────────────
        print("[Main] Entering Enhancement control loop. Press Ctrl+C to exit.\n")
        try:
            while True:
                loop_start = time.perf_counter()
                sleep_for = next_time - loop_start
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_time = time.perf_counter()
                elapsed_time = loop_start - t0

                # Interpolate reference speed
                if elapsed_time <= ref_time[0]:
                    rspd_now = ref_speed[0]
                elif elapsed_time >= ref_time[-1]:
                    rspd_now = 0.0
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))

                # Get measurements
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                e_k = rspd_now - v_meas

                # ── Enhancement Control ─────────────────────────────────────
                control_cal_start_time = time.perf_counter()
                u_total, debug_info = additive_controller.compute_control(
                    error=e_k,
                    ref_speed=rspd_now,
                    v_meas=v_meas,
                    ref_time=ref_time,
                    ref_speed_array=ref_speed,
                    elapsed_time=elapsed_time
                )
                control_cal_end_time = time.perf_counter()
                t_control = (control_cal_end_time - control_cal_start_time) * 1000

                # ── Vehicle limits and safety ────────────────────────────────
                u = float(np.clip(u_total, -30.0, +100.0))
                
                # Rate limiting (from proven baseline)
                if loop_count > 0:
                    prev_u = log_data[-1]['u']
                    u = float(np.clip(u, prev_u - max_delta, prev_u + max_delta))
                
                if latest_speed is not None and latest_speed >= 140.0:
                    u = 0.0
                    break

                # ── Send PWM to PCA9685 ──────────────────────────────────────────
                if u >= 0.0:
                    set_duty_cycle(bus, 4, 0.0)
                    set_duty_cycle(bus, 0, u)
                else:
                    set_duty_cycle(bus, 0, 0.0)
                    set_duty_cycle(bus, 4, -u)

                actual_elapsed_time = round((time.perf_counter() - loop_start)*1000,3)
                
                # ── Additive Enhancement debug printout ─────────────────────────────────────
                print(
                    "[{:.3f}] "
                    "v_ref={:6.2f}kph, "
                    "v_meas={:6.2f}kph, e={:+6.2f}kph, "
                    "u={:6.2f}%, "
                    "base={:6.2f}%, "
                    "enh={:+5.2f}%, "
                    "Kp_enh={:4.2f}x, "
                    "perf_scale={:4.2f}, "
                    "BMS_socMIN={:6.2f}%, "
                    "{}, "
                    "StartingSOC={:6.2f}%, "
                    "RunningCycle={} "
                    "t={:4.1f}ms "
                    "baseline_P={:6.2f} "
                    "baseline_I={:6.2f} "
                    "baseline_FF={:6.2f} ".format(
                        elapsed_time,
                        rspd_now,
                        v_meas, 
                        e_k,
                        u,
                        debug_info['baseline_control'],
                        debug_info['enhancement_signal'],
                        debug_info['kp_enhancement_mult'],
                        debug_info['performance_scaling'],
                        BMS_socMin,
                        'ENHANCED' if debug_info['enhancement_enabled'] else 'BASELINE',
                        SOC_CycleStarting,
                        cycle_key,
                        t_control,
                        debug_info['baseline_P'],
                        debug_info['baseline_I'],
                        debug_info['baseline_FF'],
                    )
                )

                # ── Schedule next tick ───────────────────────────────────────────
                next_time += Ts
                loop_count += 1

                # Enhanced logging with additive architecture details
                log_data.append({
                    "time": elapsed_time,
                    "v_ref": rspd_now,
                    "v_meas": v_meas,
                    "u": u,
                    "error": e_k,
                    "t_control(ms)": t_control,
                    "baseline_control": debug_info['baseline_control'],
                    "enhancement_signal": debug_info['enhancement_signal'],
                    "total_control": debug_info['total_control'],
                    "baseline_P": debug_info['baseline_P'],
                    "baseline_I": debug_info['baseline_I'],
                    "baseline_FF": debug_info['baseline_FF'],
                    "baseline_Kp": debug_info['baseline_kp'],
                    "baseline_Ki": debug_info['baseline_ki'],
                    "baseline_Kff": debug_info['baseline_kff'],
                    "kp_enhancement_mult": debug_info['kp_enhancement_mult'],
                    "ki_enhancement_mult": debug_info['ki_enhancement_mult'],
                    "performance_scaling": debug_info['performance_scaling'],
                    "enhancement_enabled": debug_info['enhancement_enabled'],
                    "tracking_success": debug_info['tracking_success'],
                    "baseline_integral_state": debug_info['baseline_integral_state'],
                    "BMS_socMin": BMS_socMin,
                    "SOC_CycleStarting": SOC_CycleStarting,
                    "actual_elapsed_time": actual_elapsed_time,
                })
                
                if BMS_socMin <= SOC_Stop:
                    break

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt detected. Exiting...")

        finally:
            # Cleanup
            for ch in range(16):
                set_duty_cycle(bus, channel=ch, percent=0.0)
            print("[Main] PWM cleaned up")

     
            # Save log data
            if log_data:
                # ═══ DTI CALCULATION SECTION ═══
                dti_metrics = {}    
                print("\n[DTI Analysis] Calculating Drive Rating Index...")
                # Convert log data to DataFrame for analysis
                df_temp = pd.DataFrame(log_data)
                # Calculate individual DTI components
                dti_metrics = calculate_dti_metrics(df_temp, ref_speed, ref_time, cycle_key)
                # Print DTI results
                print_dti_results(dti_metrics, cycle_key)

                # ==== Logging data now ========
                df = pd.DataFrame(log_data)
                df['cycle_name'] = cycle_key
                datetime_obj = datetime.now()
                df['run_datetime'] = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_str = datetime_obj.strftime("%H%M_%m%d")
                # Add DTI metrics to DataFrame for logging
                if dti_metrics:
                    for metric_key, metric_value in dti_metrics.items():
                        if metric_key != 'cycle_name':  # Avoid duplicate
                            df[f'DTI_{metric_key}'] = metric_value
                excel_filename = "{}_{}_DR_log_{}_{}_{}_Start{}%_{}_Ts{}.xlsx".format(
                    timestamp_str, 
                    "0721",  # Keep date format
                    veh_modelName, 
                    cycle_key, 
                    "Enhanced_New",  # Mark as additive version
                    SOC_CycleStarting,
                    algorithm_name, 
                    Ts)
                log_dir = os.path.join(base_folder, "Log_DriveRobot")
                os.makedirs(log_dir, exist_ok=True)
                excel_path = os.path.join(log_dir, excel_filename)
                df.to_excel(excel_path, index=False)
                print("[Main] Saved additive log to '{}'".format(excel_path))
                
                # Additive architecture performance summary
                success_rate = df['tracking_success'].sum() / len(df) * 100
                rms_error = np.sqrt(np.mean(df['error']**2))
                avg_baseline = np.mean(np.abs(df['baseline_control']))
                avg_enhancement = np.mean(np.abs(df['enhancement_signal']))
                enhancement_contribution = avg_enhancement / (avg_baseline + avg_enhancement) * 100
                
                print("[Main] Additive Enhancement Performance Summary:")
                print("  - RMS Error: {:.3f} kph".format(rms_error))
                print("  - Success Rate (<0.5kph): {:.1f}%".format(success_rate))

                # Add DTI performance summary
                if dti_metrics:
                    print("  - DTI RMSSE: {:.3f} kph (Target: <0.8)".format(dti_metrics.get('RMSSEkph', 999)))
                    print("  - DTI IWR: {:.2f}% (Target: -0.8 to +1.2)".format(dti_metrics.get('IWR', 999)))
                    print("  - DTI ASCR: {:.3f} (Target: <1.0)".format(dti_metrics.get('ASCR', 999)))
                    print("  - DTI DR: {:.3f} (Target: -0.8-1.2)".format(dti_metrics.get('DR', 999)))        
                    # Calculate DTI metrics with 1907_0721 optimized targets
                    # metrics_pass = [
                    #     dti_metrics.get('RMSSEkph', 999) < 1.5,   # Match tracking requirement
                    #     dti_metrics.get('ER', 999) < 2.0,         # Optimized from analysis
                    #     -0.8 <= dti_metrics.get('IWR', 999) <= 1.2,  # Tight efficiency target
                    #     dti_metrics.get('ASCR', 999) < 1.0,       # Ultra-tight smoothness target
                    #     dti_metrics.get('EER', 999) < 1.1,        # Achievable enhanced target
                    #     -0.8 <= dti_metrics.get('DR', 999) < 1.2          # PRIMARY GOAL: DTI < 1.2
                    # ]
                    # dti_pass_rate = sum(metrics_pass) / len(metrics_pass) * 100
                    # print("  - DTI Overall Score: {:.1f}% ({}/{} metrics passed)".format(
                    #     dti_pass_rate, sum(metrics_pass), len(metrics_pass)))
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
    print("All channels set to 0%, bus closed.")
    print("[Main] pca board PWM signal cleaned up and exited.")
    print("[Main] Cleaned up and exited.")