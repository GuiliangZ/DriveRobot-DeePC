#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRBS Data Collection Controller for DeePC Hankel Matrix Generation
Collects SISO data at different operating points using Pseudo-Random Binary Sequence (PRBS)
to build comprehensive Hankel matrices for speed-scheduled DeePC control.

Author: Generated for Drive Robot DeePC Project
Date: 2025-07-21
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import cantools
import can
from smbus2 import SMBus
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import json

# Import utilities
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils import *
from utils import compute_pid_control, get_gains_for_speed
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


class PRBSDataCollector:
    """
    Collects PRBS excitation data at different speed operating points
    for building optimal Hankel matrices for DeePC control.
    """
    
    def __init__(self, control_frequency=10):
        self.Ts = 1.0 / control_frequency
        self.control_frequency = control_frequency
        
        # PRBS parameters
        self.prbs_amplitude = 15.0  # ±15% PWM amplitude around operating point
        self.prbs_length = 1023     # Length of PRBS sequence (2^n - 1)
        self.prbs_duration = 60.0   # seconds per operating point
        
        # Operating points for WLTP cycle (kph)
        self.speed_operating_points = [
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 
            55, 60, 65, 70, 75, 80, 85, 90, 95, 100
        ]
        
        # Data storage
        self.collected_data = {}
        self.current_data = []
        
        # System state
        self.latest_speed = None
        self.latest_force = None
        self.BMS_socMin = None
        self.dyno_can_running = True
        self.veh_can_running = True
        
        # Hardware setup
        self.CP2112_BUS = 3
        self.bus = None
        
        # Initialize the Enhanced PID Controller from EnhancedPID-Controller-IWR.py
        self.controller = AdditiveEnhancementController(
            Ts=self.Ts,
            T_f=5000.0,
            FeedFwdTime=0.65,
            history_length=50,
            enable_enhancements=True  # Enable enhancements for baseline controller
        )
        
    def generate_prbs(self, n_bits=10, amplitude=1.0):
        """
        Generate Pseudo-Random Binary Sequence using linear feedback shift register.
        
        Args:
            n_bits: Number of bits for LFSR (sequence length = 2^n_bits - 1)
            amplitude: Amplitude of PRBS signal
            
        Returns:
            prbs_sequence: Array of PRBS values
        """
        # LFSR taps for different lengths (polynomial coefficients)
        taps = {
            7: [7, 6],
            8: [8, 6, 5, 4], 
            9: [9, 5],
            10: [10, 7],
            11: [11, 9],
            12: [12, 6, 4, 1]
        }
        
        if n_bits not in taps:
            raise ValueError(f"PRBS length {n_bits} not supported. Use: {list(taps.keys())}")
        
        # Initialize LFSR register
        register = [1] * n_bits
        sequence = []
        tap_positions = taps[n_bits]
        
        # Generate sequence
        for _ in range(2**n_bits - 1):
            # Calculate feedback bit
            feedback = 0
            for tap in tap_positions:
                feedback ^= register[tap-1]
            
            # Store current output
            output = 1 if register[-1] else -1
            sequence.append(output * amplitude)
            
            # Shift register and insert feedback
            register = [feedback] + register[:-1]
            
        return np.array(sequence)
    
    def pid_control(self, elapsed_time, ref_time, ref_speed, setpoint, measured_value):
        """
        Enhanced PID controller using AdditiveEnhancementController.
        
        Args:
            elapsed_time: Current elapsed time
            ref_time: Reference time array (for feed-forward)
            ref_speed: Reference speed array (for feed-forward) 
            setpoint: Desired speed (kph)
            measured_value: Current measured speed (kph)
            
        Returns:
            control_output: PWM control signal (%)
        """
        # Create simple reference arrays for the controller
        if not hasattr(self, '_ref_time_cache'):
            # Create a simple reference trajectory for feed-forward
            self._ref_time_cache = np.array([0, 1000])  # Simple 0-1000s reference
            self._ref_speed_cache = np.array([setpoint, setpoint])  # Constant setpoint
        
        # Update reference for current setpoint
        self._ref_speed_cache[:] = setpoint
        
        error = setpoint - measured_value
        
        # Use the enhanced controller
        total_control, debug_info = self.controller.compute_control(
            error=error,
            ref_speed=setpoint,
            v_meas=measured_value,
            ref_time=self._ref_time_cache,
            ref_speed_array=self._ref_speed_cache,
            elapsed_time=elapsed_time
        )
        
        return np.clip(total_control, U_MIN, U_MAX)
    
    def setup_can_communication(self):
        """Setup CAN communication for speed/force reading and vehicle SOC monitoring."""
        def dyno_can_listener():
            DYNO_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/KAVL_V3.dbc'
            DYNO_CAN_INTERFACE = 'can0'
            
            try:
                db = cantools.database.load_file(DYNO_DBC_PATH)
                speed_force_msg = db.get_message_by_name('Speed_and_Force')
                bus = can.interface.Bus(channel=DYNO_CAN_INTERFACE, interface='socketcan')
                
                print(f"[CAN] Listening on {DYNO_CAN_INTERFACE} for speed/force data...")
                
                while self.dyno_can_running:
                    msg = bus.recv(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.arbitration_id != speed_force_msg.frame_id:
                        continue
                    
                    try:
                        decoded = db.decode_message(msg.arbitration_id, msg.data)
                        self.latest_speed = decoded.get('Speed_kph', 0.0)
                        self.latest_force = decoded.get('Force_N', 0.0)
                    except:
                        continue
                        
                bus.shutdown()
            except Exception as e:
                print(f"[CAN] Dyno CAN setup failed: {e}")
        
        def veh_can_listener():
            VEH_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/vehBus.dbc'
            VEH_CAN_INTERFACE = 'can1'
            
            try:
                db = cantools.database.load_file(VEH_DBC_PATH)
                bms_soc_msg = db.get_message_by_name('BMS_socStatus')
                bus = can.interface.Bus(channel=VEH_CAN_INTERFACE, interface='socketcan')
                
                print(f"[CAN] Listening on {VEH_CAN_INTERFACE} for SOC data...")
                
                while self.veh_can_running:
                    msg = bus.recv(timeout=3.0)
                    if msg is None:
                        continue
                    if msg.arbitration_id != bms_soc_msg.frame_id:
                        continue
                    
                    try:
                        decoded = db.decode_message(msg.arbitration_id, msg.data)
                        self.BMS_socMin = decoded.get('BMS_socMin', 0.0)
                    except:
                        continue
                        
                bus.shutdown()
            except Exception as e:
                print(f"[CAN] Vehicle CAN setup failed: {e}")
        
        # Start CAN threads
        self.dyno_can_thread = threading.Thread(target=dyno_can_listener, daemon=True)
        self.veh_can_thread = threading.Thread(target=veh_can_listener, daemon=True)
        self.dyno_can_thread.start()
        self.veh_can_thread.start()
    
    def setup_pwm_hardware(self):
        """Initialize PWM hardware for vehicle control."""
        try:
            self.bus = SMBus(self.CP2112_BUS)
            print(f"[Hardware] PWM bus initialized on I2C-{self.CP2112_BUS}")
            return True
        except Exception as e:
            print(f"[Hardware] PWM setup failed: {e}")
            return False
    
    def send_pwm_signal(self, pwm_percent):
        """Send PWM signal to vehicle."""
        if self.bus is None:
            return
        
        try:
            pwm_percent = np.clip(pwm_percent, U_MIN, U_MAX)
            
            if pwm_percent >= 0.0:
                set_duty_cycle(self.bus, 4, 0.0)  # brake off
                set_duty_cycle(self.bus, 0, pwm_percent)  # accelerator
            else:
                set_duty_cycle(self.bus, 0, 0.0)  # accelerator off
                set_duty_cycle(self.bus, 4, -pwm_percent)  # brake
                
        except Exception as e:
            print(f"[Hardware] PWM send failed: {e}")
    
    def collect_data_at_operating_point(self, target_speed):
        """
        Collect PRBS excitation data at a specific operating point.
        
        Args:
            target_speed: Target speed in kph for this operating point
            
        Returns:
            dict: Collected data with timestamps, inputs, and outputs
        """
        print(f"\n[DataCollection] Starting collection at {target_speed} kph...")
        
        # Generate PRBS sequence
        prbs_sequence = self.generate_prbs(n_bits=10, amplitude=self.prbs_amplitude)
        samples_per_prbs_bit = int(self.control_frequency * 0.5)  # 0.5 seconds per bit
        
        # Extend PRBS to desired duration
        extended_prbs = np.repeat(prbs_sequence, samples_per_prbs_bit)
        total_samples = int(self.prbs_duration * self.control_frequency)
        
        # Repeat PRBS sequence to fill duration
        n_repeats = int(np.ceil(total_samples / len(extended_prbs)))
        full_prbs = np.tile(extended_prbs, n_repeats)[:total_samples]
        
        # Data storage for this operating point
        data_log = {
            'time': [],
            'target_speed': [],
            'measured_speed': [],
            'pid_control': [],
            'prbs_excitation': [],
            'total_control': [],
            'force': [],
            'soc': []
        }
        
        # Reset enhanced controller state
        self.controller.reset()
        
        # Stabilization phase - reach target speed
        print(f"[DataCollection] Stabilizing at {target_speed} kph...")
        stabilization_time = 30.0  # seconds
        stabilization_samples = int(stabilization_time * self.control_frequency)
        
        stable_count = 0
        required_stable_samples = int(5.0 * self.control_frequency)  # 5 seconds stable
        
        t0 = time.time()
        next_time = time.time()
        
        for i in range(stabilization_samples):
            loop_start = time.time()
            
            # Wait for next control cycle
            if next_time > loop_start:
                time.sleep(next_time - loop_start)
            
            elapsed = time.time() - t0
            measured_speed = self.latest_speed if self.latest_speed is not None else 0.0
            
            # PID control to reach setpoint  
            pid_output = self.pid_control(elapsed, self._ref_time_cache, self._ref_speed_cache, target_speed, measured_speed)
            self.send_pwm_signal(pid_output)
            
            # Check stability
            speed_error = abs(target_speed - measured_speed)
            if speed_error < 2.0:  # Within 2 kph
                stable_count += 1
            else:
                stable_count = 0
            
            # Early exit if stable
            if stable_count >= required_stable_samples:
                print(f"[DataCollection] Stabilized in {elapsed:.1f}s")
                break
            
            if i % (self.control_frequency * 5) == 0:  # Print every 5 seconds
                print(f"[DataCollection] Stabilizing: target={target_speed:.1f}, actual={measured_speed:.1f}, error={speed_error:.1f}")
            
            next_time += self.Ts
        
        # Data collection phase with PRBS excitation
        print(f"[DataCollection] Starting PRBS excitation for {self.prbs_duration}s...")
        
        t0 = time.time()
        next_time = time.time()
        
        for i in range(total_samples):
            loop_start = time.time()
            
            # Wait for next control cycle
            if next_time > loop_start:
                time.sleep(next_time - loop_start)
            
            elapsed = time.time() - t0
            measured_speed = self.latest_speed if self.latest_speed is not None else 0.0
            measured_force = self.latest_force if self.latest_force is not None else 0.0
            current_soc = self.BMS_socMin if self.BMS_socMin is not None else 0.0
            
            # PID control for setpoint tracking
            pid_output = self.pid_control(elapsed, self._ref_time_cache, self._ref_speed_cache, target_speed, measured_speed)
            
            # Add PRBS excitation
            prbs_excitation = full_prbs[i]
            total_control = pid_output + prbs_excitation
            
            # Send control signal
            self.send_pwm_signal(total_control)
            
            # Log data
            data_log['time'].append(elapsed)
            data_log['target_speed'].append(target_speed)
            data_log['measured_speed'].append(measured_speed)
            data_log['pid_control'].append(pid_output)
            data_log['prbs_excitation'].append(prbs_excitation)
            data_log['total_control'].append(total_control)
            data_log['force'].append(measured_force)
            data_log['soc'].append(current_soc)
            
            # Progress update
            if i % (self.control_frequency * 10) == 0:  # Every 10 seconds
                progress = (i / total_samples) * 100
                print(f"[DataCollection] Progress: {progress:.1f}% - Speed: {measured_speed:.1f} kph, Control: {total_control:.1f}%")
            
            # Safety check - stop if SOC too low
            if current_soc <= MIN_SOC_STOP:
                print(f"[DataCollection] Stopping - SOC too low: {current_soc:.1f}%")
                break
            
            next_time += self.Ts
        
        # Return to zero control
        self.send_pwm_signal(0.0)
        
        print(f"[DataCollection] Completed collection at {target_speed} kph - {len(data_log['time'])} samples")
        return data_log
    
    def run_full_data_collection(self):
        """
        Run complete data collection campaign across all operating points.
        """
        print("\n" + "="*60)
        print("PRBS DATA COLLECTION FOR DeePC HANKEL MATRICES")
        print("="*60)
        
        # Setup hardware and communication
        if not self.setup_pwm_hardware():
            print("[ERROR] Failed to initialize PWM hardware")
            return False
        
        self.setup_can_communication()
        time.sleep(2)  # Wait for CAN threads to initialize
        
        # Check initial SOC
        if self.BMS_socMin is not None and self.BMS_socMin <= MIN_SOC_STOP:
            print(f"[ERROR] Initial SOC too low: {self.BMS_socMin:.1f}%")
            return False
        
        print(f"[DataCollection] Starting collection at {len(self.speed_operating_points)} operating points")
        print(f"[DataCollection] Control frequency: {self.control_frequency} Hz")
        print(f"[DataCollection] PRBS amplitude: ±{self.prbs_amplitude}%")
        print(f"[DataCollection] Duration per point: {self.prbs_duration}s")
        print(f"[DataCollection] Initial SOC: {self.BMS_socMin:.1f}%")
        
        # Collect data at each operating point
        for i, target_speed in enumerate(self.speed_operating_points):
            print(f"\n[DataCollection] Operating point {i+1}/{len(self.speed_operating_points)}: {target_speed} kph")
            
            try:
                data = self.collect_data_at_operating_point(target_speed)
                self.collected_data[target_speed] = data
                
                # Brief rest between operating points
                print(f"[DataCollection] Resting for 10 seconds before next operating point...")
                time.sleep(10)
                
                # Check SOC after each operating point
                if self.BMS_socMin is not None and self.BMS_socMin <= MIN_SOC_STOP:
                    print(f"[DataCollection] SOC too low: {self.BMS_socMin:.1f}% - Stopping collection")
                    break
                    
            except KeyboardInterrupt:
                print(f"\n[DataCollection] Interrupted by user")
                break
            except Exception as e:
                print(f"[DataCollection] Error at {target_speed} kph: {e}")
                continue
        
        # Cleanup
        self.cleanup()
        
        # Save collected data
        self.save_collected_data()
        
        print(f"\n[DataCollection] Collection complete!")
        print(f"[DataCollection] Collected data at {len(self.collected_data)} operating points")
        
        return True
    
    def save_collected_data(self):
        """Save collected data to files for Hankel matrix generation."""
        if not self.collected_data:
            print("[DataCollection] No data to save")
            return
        
        timestamp = datetime.now().strftime("%H%M_%m%d")
        save_dir = Path("dataForHankle/PRBSCollection")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data as JSON
        json_file = save_dir / f"prbs_collection_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for speed, data in self.collected_data.items():
            json_data[str(speed)] = {
                key: (val.tolist() if isinstance(val, np.ndarray) else val)
                for key, val in data.items()
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"[DataCollection] Raw data saved to {json_file}")
        
        # Save processed data for each operating point
        for speed, data in self.collected_data.items():
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Save as Excel
            excel_file = save_dir / f"prbs_data_{speed}kph_{timestamp}.xlsx"
            df.to_excel(excel_file, index=False)
            
            # Save input-output data for Hankel matrix generation
            u_data = np.array(data['total_control'])
            y_data = np.array(data['measured_speed'])
            
            npz_file = save_dir / f"hankel_data_{speed}kph_{timestamp}.npz"
            np.savez(npz_file, ud=u_data, yd=y_data, time=np.array(data['time']))
            
            print(f"[DataCollection] Processed data saved: {excel_file} and {npz_file}")
        
        # Generate summary report
        self.generate_collection_report(save_dir, timestamp)
    
    def generate_collection_report(self, save_dir, timestamp):
        """Generate summary report of data collection."""
        report_file = save_dir / f"collection_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("PRBS DATA COLLECTION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Control Frequency: {self.control_frequency} Hz\n")
            f.write(f"PRBS Amplitude: ±{self.prbs_amplitude}%\n")
            f.write(f"Duration per Operating Point: {self.prbs_duration}s\n\n")
            
            f.write("OPERATING POINTS COLLECTED:\n")
            f.write("-"*30 + "\n")
            
            for speed in sorted(self.collected_data.keys()):
                data = self.collected_data[speed]
                n_samples = len(data['time'])
                duration = data['time'][-1] if data['time'] else 0
                avg_speed = np.mean(data['measured_speed']) if data['measured_speed'] else 0
                std_speed = np.std(data['measured_speed']) if data['measured_speed'] else 0
                
                f.write(f"Speed: {speed:3.0f} kph | ")
                f.write(f"Samples: {n_samples:4d} | ")
                f.write(f"Duration: {duration:5.1f}s | ")
                f.write(f"Avg Speed: {avg_speed:5.1f} kph | ")
                f.write(f"Std Dev: {std_speed:4.2f} kph\n")
            
            f.write(f"\nTotal Operating Points: {len(self.collected_data)}\n")
            f.write(f"Total Samples: {sum(len(data['time']) for data in self.collected_data.values())}\n")
            f.write(f"Total Duration: {sum(data['time'][-1] for data in self.collected_data.values() if data['time']):.1f}s\n")
        
        print(f"[DataCollection] Report saved to {report_file}")
    
    def cleanup(self):
        """Cleanup hardware and communication."""
        # Stop PWM
        if self.bus:
            try:
                for ch in range(16):
                    set_duty_cycle(self.bus, channel=ch, percent=0.0)
                self.bus.close()
                print("[DataCollection] PWM hardware cleaned up")
            except:
                pass
        
        # Stop CAN threads
        self.dyno_can_running = False
        self.veh_can_running = False
        
        if hasattr(self, 'dyno_can_thread'):
            self.dyno_can_thread.join(timeout=1.0)
        if hasattr(self, 'veh_can_thread'):
            self.veh_can_thread.join(timeout=1.0)
        
        print("[DataCollection] CAN communication stopped")


def main():
    """Main function to run PRBS data collection."""
    try:
        collector = PRBSDataCollector(control_frequency=10)
        success = collector.run_full_data_collection()
        
        if success:
            print("\n" + "="*60)
            print("DATA COLLECTION COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Next steps:")
            print("1. Analyze collected data quality")
            print("2. Generate optimal Hankel matrices for each operating point")
            print("3. Implement speed-scheduled Hankel matrices in DeePC controller")
        else:
            print("\n" + "="*60)
            print("DATA COLLECTION FAILED OR INCOMPLETE")
            print("="*60)
            
    except KeyboardInterrupt:
        print("\n[Main] Data collection interrupted by user")
    except Exception as e:
        print(f"\n[Main] Data collection failed: {e}")


if __name__ == "__main__":
    main()