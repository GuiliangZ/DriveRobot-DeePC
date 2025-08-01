#!/usr/bin/env python3
"""
Name: EnhancedPID-Controller-Final.py
Author: Guiliang Zheng
Date:at 22/07/2025
version: 1.0.0
Description: Additive Enhancement Controller - Preserves Baseline PID
WITH REAL-TIME IWR AND RMSSE METRICS DISPLAY

Architecture:
- Baseline PID (utils.py)
- Enhancement Layer = ADDITIVE signals on top
- Total Control = Baseline_PID + Enhancement_Signal

Enhancement Components:
1. Adaptive Gain Enhancement (based on performance)
2. Data-Driven Feedforward (from historical patterns)
3. Error Pattern Learning (predictive corrections)
4. Performance-Based Scaling (dynamic enhancement strength)

Key Principle: If enhancements are disabled, controller = exact baseline PID

ADDED FEATURES:
- Real-time IWR (Idle Waste Rate) calculation
- Real-time RMSSE (Root Mean Square Speed Error) calculation
- Checkmark display when tracking error < 0.5 kph

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
# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils import *
from smbus2 import SMBus

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
algorithm_name = "EnhancedPID-Controller-ForIWR"
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

# ──────────────────────────── Additive Enhancement Controller ──────────────────────────────
class AdditiveEnhancementController:
    def __init__(self, Ts=0.1, T_f=5000.0, FeedFwdTime=0.65, history_length=50, enable_enhancements=True):
        """
        Additive Enhancement Controller - adds enhancement signals on top of unchanged baseline PID.
        
        Core Principle: 
        Total_Control = get_baseline_PID_unchanged() + compute_enhancement_signals()
        
        Args:
            enable_enhancements: If False, controller = exact baseline PID
        """
        self.Ts = Ts
        self.T_f = T_f
        self.FeedFwdTime = FeedFwdTime
        self.enable_enhancements = enable_enhancements

        self.prev_ref_speed = 0.0
        
        # Baseline PID state (for utils_deepc.py compatibility) - NEVER MODIFIED
        self.baseline_I_state = 0.0
        self.baseline_prev_error = 0.0
        
        # Enhancement data collection
        self.error_history = deque(maxlen=history_length)
        self.speed_history = deque(maxlen=history_length)
        self.speed_derivative_history = deque(maxlen=history_length)
        self.performance_history = deque(maxlen=20)
        
        # Enhancement parameters (only active if enable_enhancements=True)
        self.adaptive_gain_range = [0.92, 1.08]    # Conservative ±8% range
        self.adaptation_rate = 0.004               # Slow, stable adaptation
        self.feedforward_scaling = 0.6             # Conservative feedforward
        self.performance_scaling_range = [0.3, 1.2] # Enhancement strength scaling
        
        # Enhancement state
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
        
        # ADDED: Real-time metrics tracking
        self.speed_error_history = deque(maxlen=500)  # For RMSSE calculation
        self.measured_speed_history = deque(maxlen=500)  # For IWR calculation
        self.reference_speed_history = deque(maxlen=500)  # For IWR calculation
        self.time_history = deque(maxlen=500)  # For IWR calculation
        
        print("[Additive Enhancement] Controller initialized:")
        print("  - Baseline PID: COMPLETELY UNCHANGED (utils_deepc.py)")
        print("  - Enhancement Mode: {}".format("ENABLED" if enable_enhancements else "DISABLED"))
        if enable_enhancements:
            print("  - Adaptive gain range: [{:.2f}, {:.2f}]".format(*self.adaptive_gain_range))
            print("  - Feedforward scaling: {:.1f}".format(self.feedforward_scaling))
            print("  - Performance scaling: [{:.1f}, {:.1f}]".format(*self.performance_scaling_range))
        print("  - Architecture: Total_Control = Baseline_PID + Enhancement_Signal")
        print("  - ADDED: Real-time IWR and RMSSE tracking")
        
    def compute_control(self, error, ref_speed, v_meas, ref_time=None, ref_speed_array=None, elapsed_time=None):
        """
        Compute total control = baseline PID (unchanged) + enhancement signals
        """
        # ═══ STEP 1: GET BASELINE PID (COMPLETELY UNCHANGED) ═══
        baseline_kp, baseline_ki, baseline_kd, baseline_kff = get_gains_for_speed(ref_speed)
        #     # ---- look ahead ----
        # t_future = elapsed_time + self.FeedFwdTime
        # if   t_future <= ref_time[0]:
        #     rspd_fut = ref_speed[0]
        # elif t_future >= ref_time[-1]:
        #     rspd_fut = 0.0
        # else:
        #     rspd_fut = float(np.interp(t_future, ref_time, ref_speed_array))
        # e_fut = (rspd_fut - v_meas) * 0.621371
        # # ---- current error in mph ----
        # e_k_mph = e_k * 0.621371
        # baseline_P = baseline_kp * e_k_mph
        # if v_meas > 0.1 and ref_speed > 0.1:
        #     self.baseline_I_state += baseline_ki * self.Ts * e_k_mph
        #     baseline_I = self.baseline_I_state
        # else:
        #     self.baseline_I_state = 0.0
        #     baseline_I   = 0.0
        # baseline_FF = baseline_kff * e_fut
        # baseline_control   = baseline_P + baseline_I + baseline_FF
        # self.baseline_prev_error = error

        # implement exact baseline PID logic
        baseline_P = baseline_kp * error

        if ref_speed > 0.01 and v_meas > 0.01:
            # Normal integral calculation with anti-windup
            self.baseline_I_state = self.baseline_I_state + baseline_ki * Ts * e_k
            baseline_I = self.baseline_I_state
        else:
            self.baseline_I_state = 0.0  # RESET INTEGRAL STATE
            baseline_I = 0.0
        
        # # Baseline integral with standard anti-windup
        # if abs(error) > 8.0:
        #     self.baseline_I_state *= 0.7
        # self.baseline_I_state += error * self.Ts
        # self.baseline_I_state = np.clip(self.baseline_I_state, -50.0, 50.0)
        # baseline_I = baseline_ki * self.baseline_I_state

        # No derivative in baseline
        baseline_D = 0.0
        # Standard baseline feedforward
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

        # ADDED: Update real-time metrics history
        self.speed_error_history.append(error)
        self.measured_speed_history.append(v_meas)
        self.reference_speed_history.append(ref_speed)
        if elapsed_time is not None:
            self.time_history.append(elapsed_time)

        # ═══ STEP 2: COMPUTE ENHANCEMENT SIGNALS ═══
        enhancement_signal = 0.0
        
        if self.enable_enhancements:
            # Update data for enhancement computation
            self.update_enhancement_data(error, ref_speed, v_meas)
            
            # Component 1: Adaptive Gain Enhancement
            adaptive_enhancement = self.compute_adaptive_gain_enhancement(error, baseline_kp, baseline_ki)
            
            # Component 2: Data-Driven Feedforward Enhancement  
            feedforward_enhancement = self.compute_feedforward_enhancement(ref_speed, ref_time, ref_speed_array, elapsed_time)
            
            # Component 3: Error Pattern Learning Enhancement
            pattern_enhancement = self.compute_pattern_enhancement(ref_speed)
            
            # Component 4: Performance-Based Scaling
            performance_scale = self.compute_performance_scaling()
            
            # Combine enhancement signals
            enhancement_signal = (adaptive_enhancement + feedforward_enhancement + pattern_enhancement) * performance_scale
            
            # Apply conservative limiting to enhancement signal
            max_enhancement = min(20.0, abs(baseline_control) * 0.3)  # Max 30% of baseline or 20%
            enhancement_signal = np.clip(enhancement_signal, -max_enhancement, max_enhancement)
            
            # Update enhancement adaptation
            self.update_enhancement_adaptation(error)
        
        # ═══ STEP 3: COMBINE SIGNALS ═══
        # Special override: When both ref_speed and measured_speed are zero, force total control to zero
        if ref_speed < 0.01 and v_meas < 0.1:
            total_control = 0.0  # Complete stop - no control input
            enhancement_signal = 0.0  # Ensure enhancement is also zero for logging
        else:
            total_control = baseline_control + enhancement_signal
        
        # ADDED: Calculate real-time metrics
        rmsse_realtime = self.calculate_realtime_rmsse()
        iwr_realtime = self.calculate_realtime_iwr()

        # Detect deceleration to zero state
        lookForwardTime_to_zero_s = 4
        resetIStateAtSpeed_kph = 4
        going_to_zero = False
        future_time_to_zero = elapsed_time + lookForwardTime_to_zero_s
        if future_time_to_zero <= ref_time[-1]:
            future_ref_to_zero = float(np.interp(future_time_to_zero, ref_time, ref_speed_array))
            going_to_zero = np.any(future_ref_to_zero <= 0.01)
        if self.prev_ref_speed > 0:
            ref_decel_rate = (ref_speed - self.prev_ref_speed) / self.Ts
            if ref_decel_rate < -0.5 and going_to_zero:
                decel_state = 'deceleratingToZero'
                if ref_speed > resetIStateAtSpeed_kph and v_meas > 0.01:
                    # Normal integral calculation with anti-windup
                    self.baseline_I_state = self.baseline_I_state + baseline_ki * Ts * e_k
                    baseline_I = self.baseline_I_state
                else:
                    self.baseline_I_state = 0.0  # RESET INTEGRAL STATE
                    baseline_I = 0.0


        self.prev_ref_speed = ref_speed
        
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
            'rmsse_realtime': rmsse_realtime,  # ADDED
            'iwr_realtime': iwr_realtime,      # ADDED
        }
        
        return total_control, debug_info
    
    def calculate_realtime_rmsse(self):
        """Calculate real-time RMSSE from recent speed errors"""
        if len(self.speed_error_history) < 10:
            return 0.0
        
        # Calculate RMSSE from recent history
        errors = np.array(list(self.speed_error_history))
        rmsse = np.sqrt(np.mean(errors**2))
        return rmsse
    
    def calculate_realtime_iwr(self):
        """Calculate real-time IWR from recent speed measurements"""
        if len(self.measured_speed_history) < 50 or len(self.time_history) < 50:
            return 0.0
        
        try:
            # Get recent data
            v_meas = np.array(list(self.measured_speed_history)[-50:])
            v_ref = np.array(list(self.reference_speed_history)[-50:])
            times = np.array(list(self.time_history)[-50:])
            
            # Calculate dt
            dt_array = np.diff(times)
            if len(dt_array) == 0 or np.mean(dt_array) == 0:
                return 0.0
            dt = np.mean(dt_array)
            
            # Convert to m/s
            v_meas_ms = v_meas / 3.6
            v_ref_ms = v_ref / 3.6
            
            # Vehicle parameters
            mass = 2200  # kg
            
            # Calculate accelerations
            accel_meas = np.gradient(v_meas_ms, dt)
            accel_ref = np.gradient(v_ref_ms, dt)
            
            # Calculate forces
            F_I_meas = mass * accel_meas
            F_I_ref = mass * accel_ref
            
            # Calculate work increments
            d_meas = v_meas_ms[:-1] * dt  # distance increments
            d_ref = v_ref_ms[:-1] * dt
            
            w_I_meas = F_I_meas[:-1] * d_meas
            w_I_ref = F_I_ref[:-1] * d_ref
            
            # Sum positive work
            IWT_meas = np.sum(w_I_meas[w_I_meas > 0])
            IWT_ref = np.sum(w_I_ref[w_I_ref > 0])
            
            # Calculate IWR
            if IWT_ref > 0:
                iwr = (IWT_meas - IWT_ref) / IWT_ref * 100
            else:
                iwr = 0.0
                
            return iwr
            
        except:
            return 0.0
    
    def update_enhancement_data(self, error, ref_speed, v_meas):
        """Update data collections for enhancement computation"""
        self.error_history.append(error)
        self.speed_history.append(ref_speed)
        self.recent_errors.append(abs(error))
        
        # Compute speed derivative
        if len(self.speed_history) >= 2:
            speed_derivative = (self.speed_history[-1] - self.speed_history[-2]) / self.Ts
            self.speed_derivative_history.append(speed_derivative)
        
        # Compute performance score
        performance_score = 1.0 / (1.0 + abs(error))
        self.recent_performance_scores.append(performance_score)
    
    def compute_adaptive_gain_enhancement(self, error, baseline_kp, baseline_ki):
        """Compute adaptive gain enhancement signals"""
        if len(self.recent_errors) < 10:
            return 0.0
        
        # Calculate additional P signal based on enhancement multiplier
        kp_enhancement_signal = baseline_kp * error * (self.kp_enhancement_mult - 1.0)
        
        # Calculate additional I signal based on enhancement multiplier  
        ki_enhancement_signal = baseline_ki * self.baseline_I_state * (self.ki_enhancement_mult - 1.0)
        
        return kp_enhancement_signal + ki_enhancement_signal
    
    def compute_feedforward_enhancement(self, ref_speed, ref_time, ref_speed_array, elapsed_time):
        """Compute data-driven feedforward enhancement"""
        if ref_time is None or ref_speed_array is None or elapsed_time is None:
            return 0.0
        
        try:
            # Get speed derivative
            future_time = elapsed_time + self.FeedFwdTime * 0.5  # Shorter horizon for enhancement
            if future_time <= ref_time[-1]:
                current_ref = float(np.interp(elapsed_time, ref_time, ref_speed_array))
                future_ref = float(np.interp(future_time, ref_time, ref_speed_array))
                speed_derivative = (future_ref - current_ref) / (self.FeedFwdTime * 0.5)
                
                # Enhanced feedforward based on learned patterns
                if abs(speed_derivative) > 0.2:
                    # Use learned feedforward gain or default
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
        # Good performance -> less enhancement needed
        # Poor performance -> more enhancement needed
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
        
        # Reset real-time metrics histories
        self.speed_error_history.clear()
        self.measured_speed_history.clear()
        self.reference_speed_history.clear()
        self.time_history.clear()
        
        # Reset enhancement adaptation states
        self.consecutive_good_count = 0
        self.consecutive_poor_count = 0
        
        # Keep learned patterns (persistent learning)
        # self.learned_feedforward_map and self.error_pattern_map are preserved
        
        print("[Additive Enhancement] Controller reset for new cycle")
        print("  - Baseline PID state: Reset to initial")
        print("  - Enhancement data: Cleared")
        print("  - Learned patterns: Preserved")

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
        enable_enhancements=True  # Set to False for exact baseline PID
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
        varname = mat_vars[0]
        ref_array = cycle_data[varname]

        # Extract reference data
        ref_time = ref_array[:, 0].astype(float).flatten()
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()
        ref_speed = ref_speed_mph * 1.60934  # Convert to kph

        # Reset for new cycle
        loop_count = 0
        additive_controller.reset()
        log_data = []
        next_time = time.perf_counter()
        t0 = time.perf_counter()
        print("\n[Main] Starting Additive Enhancement Controller")
        print("  - Architecture: Total = Baseline_PID + Enhancement_Signal")
        print("  - Baseline PID: COMPLETELY UNCHANGED (utils_deepc.py)")
        print("  - Enhancement Mode: {}".format("ENABLED" if additive_controller.enable_enhancements else "DISABLED"))

        # Real-time scheduling
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
            print("[Main] Real-time scheduling enabled")
        except:
            print("Need root for real-time scheduling")

        # ─── MAIN 10 Hz CONTROL LOOP ─────────────────────────────────────────────────
        print("[Main] Entering Additive Enhancement control loop. Press Ctrl+C to exit.\n")
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

                # ── Additive Enhancement Control ─────────────────────────────────────
                pid_start_time = time.perf_counter()
                u_total, debug_info = additive_controller.compute_control(
                    error=e_k,
                    ref_speed=rspd_now,
                    v_meas=v_meas,
                    ref_time=ref_time,
                    ref_speed_array=ref_speed,
                    elapsed_time=elapsed_time
                )
                pid_end_time = time.perf_counter()
                t_pid = (pid_end_time - pid_start_time) * 1000

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
                
                # ── Additive Enhancement debug printout WITH REAL-TIME METRICS ─────────────────────────────────────
                # ADDED: Checkmark for tracking success
                tracking_checkmark = "✓" if debug_info['tracking_success'] else " "
                
                print(
                    "[{:.3f}] "
                    "v_ref={:6.2f}kph, "
                    "v_meas={:6.2f}kph, e={:+6.2f}kph{}, "  # ADDED checkmark
                    "u={:6.2f}%, "
                    "base={:6.2f}%, "
                    "enh={:+5.2f}%, "
                    "Kp_enh={:4.2f}x, "
                    "perf_scale={:4.2f}, "
                    "RMSSE={:5.2f}kph, "  # ADDED
                    "IWR={:+6.1f}%, "      # ADDED
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
                        tracking_checkmark,  # ADDED
                        u,
                        debug_info['baseline_control'],
                        debug_info['enhancement_signal'],
                        debug_info['kp_enhancement_mult'],
                        debug_info['performance_scaling'],
                        debug_info['rmsse_realtime'],  # ADDED
                        debug_info['iwr_realtime'],     # ADDED
                        BMS_socMin,
                        'ENHANCED' if debug_info['enhancement_enabled'] else 'BASELINE',
                        SOC_CycleStarting,
                        cycle_key,
                        t_pid,
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
                    "t_pid(ms)": t_pid,
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
                    "rmsse_realtime": debug_info['rmsse_realtime'],  # ADDED
                    "iwr_realtime": debug_info['iwr_realtime'],      # ADDED
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
                excel_filename = "{}_{}_DR_log_{}_{}_Start{}%_{}_Ts{}.xlsx".format(
                    timestamp_str, 
                    "0721",  # Keep date format
                    veh_modelName, 
                    cycle_key,  
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
                print("  - Average Baseline Signal: {:.2f}%".format(avg_baseline))
                print("  - Average Enhancement Signal: {:.2f}%".format(avg_enhancement))
                print("  - Enhancement Contribution: {:.1f}%".format(enhancement_contribution))
                print("  - Architecture: Baseline_PID + Enhancement_Layer")

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
                    
                    print("  - Architecture: EnhancedPIDcontroller-ForPrecision (Simple_PID + Minimal_Enhancement)")
                
        time.sleep(5)

    # Final cleanup
    dyno_can_running = False
    veh_can_running = False
    bus.close()
    print("[Main] EnhancedPIDcontroller-ForPrecision exited successfully.")