#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Hankel DeePC Controller for Optimal WLTP Tracking
Uses pre-computed speed-scheduled Hankel matrices for optimal performance.

Enhanced version with fixed Hankel matrices collected from PRBS data at different
operating points. Provides better tracking performance by using optimal matrices
for each speed range rather than sliding window approach.

Author: Enhanced for Fixed Hankel System
Date: 2025-07-21
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
import casadi as cs
from smbus2 import SMBus

# Import DeePC utilities and solvers
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import *
from utils import compute_pid_control, get_gains_for_speed
from DeePCAcadosFixed import DeePCFixedHankelSolver
from DeePCCVXPYSolver import DeePCCVXPYWrapper
from SpeedScheduledHankel import SpeedScheduledHankel
from deepc_config import *

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
        
        # implement exact baseline PID logic
        baseline_P = baseline_kp * error

        if ref_speed > 0.01 and v_meas > 0.01:
            # Normal integral calculation with anti-windup
            self.baseline_I_state = self.baseline_I_state + baseline_ki * self.Ts * error
            baseline_I = self.baseline_I_state
        else:
            self.baseline_I_state = 0.0  # RESET INTEGRAL STATE
            baseline_I = 0.0
        
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
        future_time_to_zero = elapsed_time + lookForwardTime_to_zero_s if elapsed_time else 0
        if ref_time is not None and ref_speed_array is not None and future_time_to_zero <= ref_time[-1]:
            future_ref_to_zero = float(np.interp(future_time_to_zero, ref_time, ref_speed_array))
            going_to_zero = np.any(future_ref_to_zero <= 0.01)
        if self.prev_ref_speed > 0:
            ref_decel_rate = (ref_speed - self.prev_ref_speed) / self.Ts
            if ref_decel_rate < -0.5 and going_to_zero:
                decel_state = 'deceleratingToZero'
                if ref_speed > resetIStateAtSpeed_kph and v_meas > 0.01:
                    # Normal integral calculation with anti-windup
                    self.baseline_I_state = self.baseline_I_state + baseline_ki * self.Ts * error
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

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
algorithm_name = "PID_DeePC_Fixed_10Hz"
latest_speed = None
latest_force = None
dyno_can_running = True
veh_can_running = True
BMS_socMin = None
CP2112_BUS = 3

# Global variable for PWM bus
bus = None

def dyno_can_listener_thread(dbc_path: str, can_iface: str):
    """CAN listener for speed and force data."""
    global latest_speed, latest_force, dyno_can_running

    try:
        db = cantools.database.load_file(dbc_path)
        speed_force_msg = db.get_message_by_name('Speed_and_Force')
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
        
        print(f"[CAN⋅Thread] Listening on {can_iface} for ID=0x{speed_force_msg.frame_id:03X}…")
        
        while dyno_can_running:
            msg = bus.recv(timeout=1.0)
            if msg is None:
                continue
            if msg.arbitration_id != speed_force_msg.frame_id:
                continue
                
            try:
                decoded = db.decode_message(msg.arbitration_id, msg.data)
                s = decoded.get('Speed_kph')
                f = decoded.get('Force_N')
                if s is not None:
                    latest_speed = float(s) if abs(s) > 0.1 else 0.0
                if f is not None:
                    latest_force = float(f)
            except:
                continue

        bus.shutdown()
        
    except Exception as e:
        print(f"[CAN⋅Thread] Error: {e}")

def veh_can_listener_thread(dbc_path: str, can_iface: str):
    """CAN listener for vehicle SOC data."""
    global BMS_socMin, veh_can_running

    try:
        db = cantools.database.load_file(dbc_path)
        bms_soc_msg = db.get_message_by_name('BMS_socStatus')
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
        
        print(f"[CAN⋅Thread] Listening on {can_iface} for SOC data…")
        
        while veh_can_running:
            msg = bus.recv(timeout=3.0)
            if msg is None:
                continue
            if msg.arbitration_id != bms_soc_msg.frame_id:
                continue
                
            try:
                decoded = db.decode_message(msg.arbitration_id, msg.data)
                BMS_socMin = decoded.get('BMS_socMin')
                if BMS_socMin is not None:
                    BMS_socMin = float(BMS_socMin)
            except:
                continue

        bus.shutdown()
        
    except Exception as e:
        print(f"[CAN⋅Thread] Vehicle CAN error: {e}")


class FixedHankelDeePCController:
    """
    Enhanced DeePC controller using fixed pre-computed Hankel matrices
    optimized for different speed operating points.
    """
    
    def __init__(self, hankel_data_file=None):
        # System parameters
        self.Ts = 1.0 / CONTROL_FREQUENCY
        self.control_frequency = CONTROL_FREQUENCY
        self.max_delta = 80.0  # Max % change per cycle
        self.T_f = 5000.0      # Derivative filter coefficient
        
        # Load configuration
        config_errors = validate_config()
        if config_errors:
            raise RuntimeError(f"Configuration validation failed: {config_errors}")
        
        # Find latest Hankel matrix file if not provided
        if hankel_data_file is None:
            hankel_dir = Path("dataForHankle/OptimizedMatrices")
            pickle_files = list(hankel_dir.glob("complete_hankel_collection_*.pkl"))
            if not pickle_files:
                raise FileNotFoundError("No Hankel matrix files found. Run data collection and analysis first.")
            hankel_data_file = str(max(pickle_files, key=lambda x: x.stat().st_mtime))
            print(f"[Controller] Using Hankel matrices: {hankel_data_file}")
        
        # Define system constraints
        self.ineqconidx = {'u': [0], 'y': [0]}
        self.ineqconbd = {
            'lbu': np.array([U_MIN]), 
            'ubu': np.array([U_MAX]),
            'lby': np.array([Y_MIN]), 
            'uby': np.array([Y_MAX])
        }
        
        # Add rate constraints if enabled
        if ENABLE_RATE_LIMITING:
            max_delta_per_cycle = MAX_CONTROL_RATE * self.Ts
            self.ineqconidx['du'] = [0]
            self.ineqconbd['lbdu'] = np.array([-max_delta_per_cycle])
            self.ineqconbd['ubdu'] = np.array([max_delta_per_cycle])
        
        # Initialize Fixed Hankel DeePC solver
        print("[Controller] Initializing Fixed Hankel DeePC solver...")
        self.dpc_solver = DeePCFixedHankelSolver(
            u_dim=1, y_dim=1,
            hankel_data_file=hankel_data_file,
            ineqconidx=self.ineqconidx,
            ineqconbd=self.ineqconbd
        )
        
        # Initialize Acados solver
        recompile = not Path("DeePC_fixed_acados_ocp.json").exists()
        self.dpc_solver.init_acados_solver(recompile_solver=recompile)
        
        # Get system information
        self.system_info = self.dpc_solver.hankel_scheduler.get_system_info()
        print(f"[Controller] System ready with {self.system_info['num_operating_points']} operating points")
        print(f"[Controller] Speed range: {self.system_info['speed_range'][0]:.0f} - {self.system_info['speed_range'][1]:.0f} kph")
        
        # Performance monitoring
        self.solve_time_history = deque(maxlen=100)
        self.success_rate_history = deque(maxlen=100)
        self.matrix_update_count = 0
        
        # Initialize Enhanced PID Controller (fallback) - using AdditiveEnhancementController
        self.fallback_controller = AdditiveEnhancementController(
            Ts=self.Ts,
            T_f=5000.0,
            FeedFwdTime=0.65,
            history_length=50,
            enable_enhancements=True  # Enable enhancements for fallback controller
        )
    
    def apply_baseline_pid_control(self, elapsed_time, ref_time, ref_speed, v_meas, e_k):
        """Apply enhanced PID controller using AdditiveEnhancementController."""
        # Get reference speed at current time
        if ref_time is not None and ref_speed is not None:
            ref_speed_value = float(np.interp(elapsed_time, ref_time, ref_speed))
        else:
            ref_speed_value = 0.0
            
        # Use the enhanced controller
        total_control, debug_info = self.fallback_controller.compute_control(
            error=e_k,
            ref_speed=ref_speed_value,
            v_meas=v_meas,
            ref_time=ref_time,
            ref_speed_array=ref_speed,
            elapsed_time=elapsed_time
        )
        
        # Extract PID components for compatibility with existing code
        P_term = debug_info.get('baseline_P', 0.0)
        I_out = debug_info.get('baseline_I', 0.0)
        D_term = 0.0  # No D term in baseline
        e_k_mph = e_k * 0.621371  # Convert to mph
        
        return total_control, P_term, I_out, D_term, e_k_mph
    
    def create_weighting_matrices(self, THorizon, deepc_params):
        """Create time-varying weighting matrices."""
        # Extract parameters
        Q_val = deepc_params.Q_val
        R_val = deepc_params.R_val
        decay_rate_q = deepc_params.decay_rate_q
        decay_rate_r = deepc_params.decay_rate_r
        lambda_g_val = deepc_params.lambda_g_val
        lambda_y_val = deepc_params.lambda_y_val
        lambda_u_val = deepc_params.lambda_u_val
        
        # Create time-varying weights
        q_weights = Q_val * np.exp(-decay_rate_q * np.arange(THorizon))
        r_weights = R_val * np.exp(-decay_rate_r * np.arange(THorizon))
        
        Q = np.diag(q_weights)
        R = np.diag(r_weights)
        
        return Q, R, lambda_g_val, lambda_y_val, lambda_u_val
    
    def run_cycle(self, cycle_key, cycle_data, veh_modelName):
        """Run a single drive cycle with Fixed Hankel DeePC control."""
        print(f"\n[Controller] Starting cycle '{cycle_key}' with Fixed Hankel DeePC")
        
        # Extract cycle data
        mat_vars = [k for k in cycle_data.keys() if not k.startswith("__")]
        if len(mat_vars) != 1:
            raise RuntimeError(f"Expected exactly one variable in '{cycle_key}', found {mat_vars}")
        
        varname = mat_vars[0]
        ref_array = cycle_data[varname]
        ref_time = ref_array[:, 0].astype(float).flatten()
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()
        ref_speed = ref_speed_mph * 1.60934  # Convert to kph
        
        # Initialize control parameters
        deepc_params = MANUAL_DEEPC_PARAMS
        
        # Initialize state variables
        loop_count = 0
        prev_error = 0.0
        I_state = 0.0
        D_f_state = 0.0
        u_prev = 0.0
        g_prev = None
        
        # Initialize histories for state estimation
        Tini = self.dpc_solver.Tini_max  # Use maximum Tini for consistency
        u_history = deque([0.0] * Tini, maxlen=Tini)
        spd_history = deque([0.0] * Tini, maxlen=Tini)
        
        # Performance tracking
        deepc_activations = 0
        pid_activations = 0
        matrix_updates = 0
        last_matrix_update_speed = None
        
        # Data logging
        log_data = []
        
        # Timing setup
        next_time = time.perf_counter()
        t0 = time.perf_counter()
        
        print(f"[Controller] Starting 10Hz control loop for {cycle_key}")
        print(f"[Controller] Cycle duration: {ref_time[-1]:.1f}s")
        
        # Real-time scheduling (if possible)
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(99))
        except:
            pass
        
        try:
            while True:
                loop_start = time.perf_counter()
                sleep_for = next_time - loop_start
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_time = time.perf_counter()
                
                elapsed_time = loop_start - t0
                
                # Get reference speed
                if elapsed_time <= ref_time[0]:
                    rspd_now = ref_speed[0]
                elif elapsed_time >= ref_time[-1]:
                    rspd_now = 0.0
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))
                
                # Get measured values
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                current_soc = BMS_socMin if BMS_socMin is not None else 100.0
                
                # Compute tracking error
                e_k = rspd_now - v_meas
                
                # Check if DeePC matrices need update
                matrix_updated = self.dpc_solver.update_matrices_for_speed(v_meas)
                if matrix_updated:
                    matrix_updates += 1
                    last_matrix_update_speed = v_meas
                
                # Get current matrix information
                matrix_info = self.dpc_solver.get_current_matrices_info()
                current_params = matrix_info['params']
                active_Tini = current_params['Tini']
                active_THorizon = current_params['THorizon']
                
                # Build reference horizon
                t_future = elapsed_time + self.Ts * np.arange(active_THorizon)
                if t_future[-1] >= ref_time[-1]:
                    valid_mask = t_future <= ref_time[-1]
                    active_THorizon = int(valid_mask.sum())
                    t_future = t_future[valid_mask]
                
                if active_THorizon <= 0:
                    break
                
                ref_horizon_speed = np.interp(t_future, ref_time, ref_speed)
                ref_horizon_speed = ref_horizon_speed.reshape(-1, 1)
                
                # Prepare DeePC inputs
                u_init = np.array(list(u_history)[-active_Tini:]).reshape(-1, 1)
                y_init = np.array(list(spd_history)[-active_Tini:]).reshape(-1, 1)
                
                # Create weighting matrices
                Q, R, lambda_g_val, lambda_y_val, lambda_u_val = self.create_weighting_matrices(
                    active_THorizon, deepc_params
                )
                
                # Regularization matrices
                lambda_g = np.diag(np.tile(lambda_g_val, current_params['g_dim']))
                lambda_y = np.diag(np.tile(lambda_y_val, active_Tini))
                lambda_u = np.diag(np.tile(lambda_u_val, active_Tini))
                
                # Attempt DeePC solve
                deepc_success = False
                cost = float('inf')
                solve_time = 0.0
                
                try:
                    u_opt, g_opt, solve_time, feasible, cost = self.dpc_solver.solve_step(
                        u_init, y_init, ref_horizon_speed, v_meas,
                        Q, R, lambda_g, lambda_y, lambda_u, g_prev
                    )
                    
                    # Update performance tracking
                    self.solve_time_history.append(solve_time)
                    self.success_rate_history.append(feasible)
                    
                    # Check if solution is acceptable
                    max_allowed_time = self.Ts * 1000 * MAX_SOLVE_TIME_RATIO
                    if feasible and cost >= 0 and solve_time < max_allowed_time:
                        u_unclamped = u_opt[0]
                        deepc_success = True
                        deepc_activations += 1
                        g_prev = g_opt
                        controller_type = "FixedDeePC"
                    else:
                        g_prev = None
                        
                except Exception as e:
                    print(f"[Controller] DeePC solve error: {e}")
                
                # Fallback to PID if DeePC failed
                if not deepc_success:
                    u_pid, P_term, I_out, D_term, e_k_mph = self.apply_baseline_pid_control(
                        elapsed_time, ref_time, ref_speed, v_meas, e_k
                    )
                    u_unclamped = u_pid
                    pid_activations += 1
                    controller_type = "PID"
                
                # Apply control limits and rate limiting
                u = float(np.clip(u_unclamped, U_MIN, U_MAX))
                
                if ENABLE_RATE_LIMITING:
                    max_delta_safety = MAX_CONTROL_RATE * self.Ts * 1.2
                    u = float(np.clip(u, u_prev - max_delta_safety, u_prev + max_delta_safety))
                
                # Emergency stop
                if v_meas >= EMERGENCY_SPEED_LIMIT:
                    u = 0.0
                    print(f"[SAFETY] Emergency stop: {v_meas:.1f} >= {EMERGENCY_SPEED_LIMIT:.1f} kph")
                    break
                
                # Send PWM control signal
                if u >= 0.0:
                    set_duty_cycle(bus, 4, 0.0)  # brake off
                    set_duty_cycle(bus, 0, u)    # accelerator
                else:
                    set_duty_cycle(bus, 0, 0.0)  # accelerator off
                    set_duty_cycle(bus, 4, -u)   # brake
                
                # Update histories
                u_history.append(u)
                spd_history.append(v_meas)
                u_prev = u
                
                # Performance monitoring
                actual_elapsed = (time.perf_counter() - loop_start) * 1000
                actual_freq = 1000 / actual_elapsed if actual_elapsed > 0 else 0
                
                # Status output
                if loop_count % (CONTROL_FREQUENCY * 2) == 0:  # Every 2 seconds
                    avg_solve_time = np.mean(self.solve_time_history) if self.solve_time_history else 0
                    success_rate = np.mean(self.success_rate_history) if self.success_rate_history else 0
                    
                    print(f"[{elapsed_time:6.1f}s] "
                          f"v_ref={rspd_now:5.1f}, v_meas={v_meas:5.1f}, e={e_k:+5.1f} kph | "
                          f"u={u:6.1f}%, {controller_type} | "
                          f"solve_t={solve_time:4.1f}ms, freq={actual_freq:4.1f}Hz | "
                          f"SOC={current_soc:4.1f}% | "
                          f"Matrix: {matrix_info['method']} @ {matrix_info.get('source_speed', 'N/A')}")
                
                # Data logging
                log_data.append({
                    "time": elapsed_time,
                    "v_ref": rspd_now,
                    "v_meas": v_meas,
                    "u": u,
                    "error": e_k,
                    "controller_type": controller_type,
                    "solve_time_ms": solve_time,
                    "cost": cost,
                    "BMS_socMin": current_soc,
                    "matrix_method": matrix_info['method'],
                    "matrix_source_speed": matrix_info.get('source_speed', 0),
                    "matrix_extrapolation": matrix_info.get('extrapolation', 0),
                    "active_Tini": active_Tini,
                    "active_THorizon": active_THorizon,
                    "actual_frequency": actual_freq,
                    "deepc_success": deepc_success,
                    "matrix_updated": matrix_updated
                })
                
                # Safety check
                if current_soc <= MIN_SOC_STOP:
                    print(f"[Controller] SOC too low: {current_soc:.1f}% - Stopping")
                    break
                
                # Update loop timing
                next_time += self.Ts
                loop_count += 1
                
        except KeyboardInterrupt:
            print("\n[Controller] Interrupted by user")
        
        finally:
            # Stop PWM
            for ch in range(16):
                set_duty_cycle(bus, channel=ch, percent=0.0)
        
        # Performance summary
        total_activations = deepc_activations + pid_activations
        deepc_percentage = (deepc_activations / total_activations * 100) if total_activations > 0 else 0
        
        print(f"\n[Controller] Cycle '{cycle_key}' completed:")
        print(f"  Total cycles: {loop_count}")
        print(f"  DeePC activations: {deepc_activations} ({deepc_percentage:.1f}%)")
        print(f"  PID fallback activations: {pid_activations}")
        print(f"  Matrix updates: {matrix_updates}")
        print(f"  Average solve time: {np.mean(self.solve_time_history):.2f}ms")
        print(f"  Success rate: {np.mean(self.success_rate_history)*100:.1f}%")
        
        # Save log data
        if log_data:
            self.save_log_data(log_data, cycle_key, veh_modelName, deepc_params)
        
        return {
            'total_cycles': loop_count,
            'deepc_activations': deepc_activations,
            'pid_activations': pid_activations,
            'matrix_updates': matrix_updates,
            'avg_solve_time': np.mean(self.solve_time_history),
            'success_rate': np.mean(self.success_rate_history)
        }
    
    def save_log_data(self, log_data, cycle_key, veh_modelName, deepc_params):
        """Save cycle log data to Excel file."""
        df = pd.DataFrame(log_data)
        df['cycle_name'] = cycle_key
        df['algorithm'] = algorithm_name
        
        # Add parameter information
        df['Q_val'] = deepc_params.Q_val
        df['R_val'] = deepc_params.R_val
        df['lambda_g'] = deepc_params.lambda_g_val
        df['lambda_y'] = deepc_params.lambda_y_val
        df['lambda_u'] = deepc_params.lambda_u_val
        
        # Create filename
        timestamp = datetime.now().strftime("%H%M_%m%d")
        SOC_start = BMS_socMin if BMS_socMin else 0
        filename = (f"{timestamp}_DR_log_{veh_modelName}_{cycle_key}_"
                   f"Start{SOC_start:.1f}%_{algorithm_name}_Ts{self.Ts:.3f}_"
                   f"Q{deepc_params.Q_val}_R{deepc_params.R_val}_FixedHankel.xlsx")
        
        # Save file
        log_dir = Path("Log_DriveRobot")
        log_dir.mkdir(exist_ok=True)
        excel_path = log_dir / filename
        
        df.to_excel(excel_path, index=False)
        print(f"[Controller] Log saved: {filename}")


def main():
    """Main function to run Fixed Hankel DeePC controller."""
    global dyno_can_running, veh_can_running, bus
    
    print("\n" + "="*80)
    print("FIXED HANKEL DeePC CONTROLLER FOR OPTIMAL WLTP TRACKING")
    print("="*80)
    
    # Initialize controller
    try:
        controller = FixedHankelDeePCController()
    except Exception as e:
        print("[ERROR] Controller initialization failed: {}".format(e))
        return
    
    # Setup hardware
    print("[Main] Setting up hardware...")
    bus = SMBus(CP2112_BUS)
    
    # Start CAN threads
    DYNO_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/KAVL_V3.dbc'
    DYNO_CAN_INTERFACE = 'can0'
    VEH_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/vehBus.dbc'
    VEH_CAN_INTERFACE = 'can1'
    
    if dyno_can_running:
        dyno_thread = threading.Thread(
            target=dyno_can_listener_thread,
            args=(DYNO_DBC_PATH, DYNO_CAN_INTERFACE),
            daemon=True
        )
        dyno_thread.start()
    
    if veh_can_running:
        veh_thread = threading.Thread(
            target=veh_can_listener_thread,
            args=(VEH_DBC_PATH, VEH_CAN_INTERFACE),
            daemon=True
        )
        veh_thread.start()
    
    # Wait for CAN initialization
    time.sleep(2)
    
    # Check initial SOC
    if BMS_socMin is not None and BMS_socMin <= MIN_SOC_STOP:
        print(f"[ERROR] Initial SOC too low: {BMS_socMin:.1f}%")
        return
    
    # Load and select drive cycles
    print("[Main] Loading drive cycles...")
    all_cycles = load_drivecycle_mat_files("")
    cycle_keys = choose_cycle_key(all_cycles)
    veh_modelName = choose_vehicleModelName()
    
    print(f"[Main] Selected {len(cycle_keys)} cycles for testing")
    print(f"[Main] Vehicle model: {veh_modelName}")
    print(f"[Main] Initial SOC: {BMS_socMin:.1f}%")
    print(f"[Main] System will stop at SOC: {MIN_SOC_STOP:.1f}%")
    
    # Run cycles
    results = {}
    for i, cycle_key in enumerate(cycle_keys):
        print(f"\n[Main] === Cycle {i+1}/{len(cycle_keys)}: {cycle_key} ===")
        
        # Check SOC before starting cycle
        if BMS_socMin is not None and BMS_socMin <= MIN_SOC_STOP:
            print(f"[Main] SOC too low ({BMS_socMin:.1f}%) - Stopping tests")
            break
        
        try:
            cycle_data = all_cycles[cycle_key]
            cycle_results = controller.run_cycle(cycle_key, cycle_data, veh_modelName)
            results[cycle_key] = cycle_results
            
            # Brief rest between cycles
            print(f"[Main] Resting 10 seconds before next cycle...")
            time.sleep(10)
            
        except Exception as e:
            print(f"[Main] Cycle {cycle_key} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cleanup
    dyno_can_running = False
    veh_can_running = False
    
    try:
        for ch in range(16):
            set_duty_cycle(bus, channel=ch, percent=0.0)
        bus.close()
    except:
        pass
    
    # Final summary
    print(f"\n{'='*80}")
    print("FIXED HANKEL DeePC TESTING COMPLETED")
    print(f"{'='*80}")
    print(f"Cycles completed: {len(results)}")
    
    if results:
        total_cycles = sum(r['total_cycles'] for r in results.values())
        total_deepc = sum(r['deepc_activations'] for r in results.values())
        avg_solve_time = np.mean([r['avg_solve_time'] for r in results.values()])
        avg_success_rate = np.mean([r['success_rate'] for r in results.values()])
        
        print(f"Overall Statistics:")
        print(f"  Total control cycles: {total_cycles}")
        print(f"  DeePC success rate: {total_deepc/total_cycles*100:.1f}%")
        print(f"  Average solve time: {avg_solve_time:.2f}ms")
        print(f"  Average success rate: {avg_success_rate*100:.1f}%")
        
        for cycle, result in results.items():
            deepc_pct = result['deepc_activations']/result['total_cycles']*100
            print(f"  {cycle}: {deepc_pct:.1f}% DeePC, {result['matrix_updates']} matrix updates")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()