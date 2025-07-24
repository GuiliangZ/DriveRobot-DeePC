#!/usr/bin/env python3
"""
AdaptiveOnlineTuner.py
Context-aware online parameter tuning for DeePC with dynamic Hankel matrices
Author: Claude AI Assistant
Date: 2025-01-21

This module addresses the challenge of parameter tuning when both reference
speed and Hankel matrices are constantly changing due to sliding window operation.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from DeePCParameterTuner import DeePCParameters

class OperatingCondition(NamedTuple):
    """Represents an operating condition for parameter tuning"""
    speed_range: Tuple[float, float]  # (min_speed, max_speed)
    acceleration_type: str            # 'accel', 'decel', 'cruise'
    gradient_level: str               # 'low', 'medium', 'high'
    
    def __hash__(self):
        return hash((self.speed_range, self.acceleration_type, self.gradient_level))

@dataclass
class ContextualPerformance:
    """Performance metrics for a specific operating condition"""
    condition: OperatingCondition
    tracking_errors: List[float]
    control_efforts: List[float]
    solve_times: List[float]
    success_flags: List[bool]
    reference_speeds: List[float]
    timestamps: List[float]
    
    # Computed metrics
    tracking_rms: float = 0.0
    control_smoothness: float = 0.0
    success_rate: float = 0.0
    sample_count: int = 0
    
    def update_metrics(self):
        """Update computed metrics from raw data"""
        if len(self.tracking_errors) > 0:
            self.tracking_rms = np.sqrt(np.mean(np.array(self.tracking_errors)**2))
            
        if len(self.control_efforts) > 1:
            control_changes = np.diff(self.control_efforts)
            self.control_smoothness = np.var(control_changes)
            
        if len(self.success_flags) > 0:
            self.success_rate = np.mean(self.success_flags)
            
        self.sample_count = len(self.tracking_errors)

@dataclass
class AdaptiveTuningConfig:
    """Configuration for adaptive online tuning"""
    # Context classification parameters
    speed_bins: List[Tuple[float, float]] = None  # [(0,20), (20,50), (50,80), (80,120)]
    accel_threshold: float = 0.5                  # kph/s threshold for accel/decel classification
    gradient_thresholds: Tuple[float, float] = (0.2, 0.8)  # Low/medium/high speed change rates
    
    # Tuning parameters
    min_samples_per_condition: int = 50           # Minimum samples before tuning in a condition
    max_samples_per_condition: int = 200          # Maximum samples to keep per condition
    similarity_threshold: float = 0.7             # Minimum similarity to reuse parameters
    
    # Parameter bounds (same as before but can be condition-specific)
    Q_bounds: Tuple[float, float] = (100.0, 800.0)
    R_bounds: Tuple[float, float] = (0.02, 0.5)
    lambda_g_bounds: Tuple[float, float] = (20.0, 200.0)
    lambda_y_bounds: Tuple[float, float] = (5.0, 50.0)
    lambda_u_bounds: Tuple[float, float] = (5.0, 50.0)
    decay_q_bounds: Tuple[float, float] = (0.6, 0.95)
    decay_r_bounds: Tuple[float, float] = (0.05, 0.3)
    
    # Adaptation parameters
    learning_rate: float = 0.1
    momentum: float = 0.3
    exploration_rate: float = 0.05
    
    def __post_init__(self):
        if self.speed_bins is None:
            self.speed_bins = [(0, 25), (25, 50), (50, 80), (80, 120)]

class AdaptiveOnlineTuner:
    """
    Adaptive online parameter tuner that accounts for changing operating conditions
    """
    
    def __init__(self, config: AdaptiveTuningConfig, initial_params: DeePCParameters):
        self.config = config
        self.initial_params = initial_params
        
        # Fixed structure parameters (never change)
        self.fixed_Tini = initial_params.Tini
        self.fixed_THorizon = initial_params.THorizon
        self.fixed_hankel_size = initial_params.hankel_subB_size
        
        # Context-aware performance tracking
        self.condition_performance: Dict[OperatingCondition, ContextualPerformance] = {}
        self.condition_parameters: Dict[OperatingCondition, DeePCParameters] = {}
        
        # Current state
        self.current_condition: Optional[OperatingCondition] = None
        self.current_params = initial_params
        self.current_buffer = []  # Buffer for current condition data
        
        # Adaptive learning state
        self.condition_visits = defaultdict(int)
        self.global_best_score = float('inf')
        self.global_best_params = initial_params
        
        # Recent history for context classification
        self.speed_history = deque(maxlen=20)  # 2 seconds at 10Hz
        self.accel_history = deque(maxlen=20)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Tuning state
        self.is_active = False
        self.total_updates = 0
        
        print(f"[AdaptiveTuner] Initialized with {len(config.speed_bins)} speed bins")
        print(f"  Fixed structure: Tini={self.fixed_Tini}, THorizon={self.fixed_THorizon}")
    
    def start_tuning(self):
        """Start adaptive online tuning"""
        with self.lock:
            self.is_active = True
        print("[AdaptiveTuner] Started adaptive online tuning")
    
    def stop_tuning(self):
        """Stop adaptive online tuning"""
        with self.lock:
            self.is_active = False
        print("[AdaptiveTuner] Stopped adaptive online tuning")
    
    def classify_operating_condition(self, ref_speed: float, measured_speed: float, 
                                   dt: float = 0.1) -> OperatingCondition:
        """
        Classify current operating condition based on speed profile characteristics
        """
        # Update speed and acceleration history
        self.speed_history.append(measured_speed)
        
        if len(self.speed_history) >= 2:
            accel = (self.speed_history[-1] - self.speed_history[-2]) / dt
            self.accel_history.append(accel)
        
        # Determine speed range
        speed_range = None
        for speed_bin in self.config.speed_bins:
            if speed_bin[0] <= measured_speed < speed_bin[1]:
                speed_range = speed_bin
                break
        if speed_range is None:
            # Handle edge case - use closest bin
            speed_range = self.config.speed_bins[-1]
        
        # Determine acceleration type
        if len(self.accel_history) >= 5:
            recent_accel = np.mean(list(self.accel_history)[-5:])  # Average over 0.5s
            if recent_accel > self.config.accel_threshold:
                accel_type = 'accel'
            elif recent_accel < -self.config.accel_threshold:
                accel_type = 'decel'
            else:
                accel_type = 'cruise'
        else:
            accel_type = 'cruise'
        
        # Determine gradient level (speed change rate relative to reference)
        if len(self.speed_history) >= 10:
            speed_variation = np.std(list(self.speed_history)[-10:])
            ref_speed_avg = np.mean([ref_speed] * 10)  # Use current ref as approximation
            if ref_speed_avg > 0:
                normalized_variation = speed_variation / ref_speed_avg
                if normalized_variation < self.config.gradient_thresholds[0]:
                    gradient_level = 'low'
                elif normalized_variation < self.config.gradient_thresholds[1]:
                    gradient_level = 'medium'
                else:
                    gradient_level = 'high'
            else:
                gradient_level = 'low'
        else:
            gradient_level = 'low'
        
        return OperatingCondition(speed_range, accel_type, gradient_level)
    
    def record_performance(self, ref_speed: float, measured_speed: float, 
                         tracking_error: float, control_effort: float,
                         solve_time: float, success: bool, cost: float):
        """Record performance data with operating condition context"""
        if not self.is_active:
            return
        
        with self.lock:
            # Classify current operating condition
            condition = self.classify_operating_condition(ref_speed, measured_speed)
            
            # Check if we've transitioned to a new condition
            if self.current_condition != condition:
                self._handle_condition_transition(condition)
            
            # Record data for current condition
            if condition not in self.condition_performance:
                self.condition_performance[condition] = ContextualPerformance(
                    condition=condition,
                    tracking_errors=[], control_efforts=[], solve_times=[],
                    success_flags=[], reference_speeds=[], timestamps=[]
                )
            
            perf = self.condition_performance[condition]
            perf.tracking_errors.append(abs(tracking_error))
            perf.control_efforts.append(control_effort)
            perf.solve_times.append(solve_time)
            perf.success_flags.append(success)
            perf.reference_speeds.append(ref_speed)
            perf.timestamps.append(time.time())
            
            # Limit buffer size to prevent memory growth
            if len(perf.tracking_errors) > self.config.max_samples_per_condition:
                # Keep most recent samples
                keep_idx = len(perf.tracking_errors) - self.config.max_samples_per_condition
                perf.tracking_errors = perf.tracking_errors[keep_idx:]
                perf.control_efforts = perf.control_efforts[keep_idx:]
                perf.solve_times = perf.solve_times[keep_idx:]
                perf.success_flags = perf.success_flags[keep_idx:]
                perf.reference_speeds = perf.reference_speeds[keep_idx:]
                perf.timestamps = perf.timestamps[keep_idx:]
    
    def _handle_condition_transition(self, new_condition: OperatingCondition):
        """Handle transition to a new operating condition"""
        print(f"[AdaptiveTuner] Condition change: {self.current_condition} -> {new_condition}")
        
        # Update previous condition metrics if we had one
        if self.current_condition is not None:
            self._finalize_condition_evaluation(self.current_condition)
        
        # Set new condition and load appropriate parameters
        self.current_condition = new_condition
        self.condition_visits[new_condition] += 1
        
        # Load parameters for new condition
        if new_condition in self.condition_parameters:
            # Use previously tuned parameters for this condition
            self.current_params = self.condition_parameters[new_condition]
            print(f"[AdaptiveTuner] Loaded tuned parameters for condition {new_condition}")
        else:
            # Find similar condition or use defaults
            similar_params = self._find_similar_condition_parameters(new_condition)
            if similar_params is not None:
                self.current_params = similar_params
                print(f"[AdaptiveTuner] Using similar condition parameters")
            else:
                self.current_params = self.initial_params
                print(f"[AdaptiveTuner] Using initial parameters for new condition")
    
    def _finalize_condition_evaluation(self, condition: OperatingCondition):
        """Finalize evaluation for a condition and update parameters if needed"""
        if condition not in self.condition_performance:
            return
        
        perf = self.condition_performance[condition]
        
        # Only tune if we have sufficient data
        if len(perf.tracking_errors) < self.config.min_samples_per_condition:
            return
        
        perf.update_metrics()
        
        # Calculate performance score
        score = self._calculate_contextual_score(perf)
        
        # Update parameters if this is better than current best for this condition
        if (condition not in self.condition_parameters or
            score < self._get_condition_best_score(condition)):
            
            # Create optimized parameters for this condition
            optimized_params = self._optimize_for_condition(condition, perf)
            self.condition_parameters[condition] = optimized_params
            
            print(f"[AdaptiveTuner] Updated parameters for {condition}:")
            print(f"  Tracking RMS: {perf.tracking_rms:.3f} kph")
            print(f"  Success rate: {perf.success_rate:.1%}")
            print(f"  Score: {score:.3f}")
            
            # Update global best if applicable
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_params = optimized_params
                print(f"[AdaptiveTuner] New global best score: {score:.3f}")
        
        self.total_updates += 1
    
    def _find_similar_condition_parameters(self, condition: OperatingCondition) -> Optional[DeePCParameters]:
        """Find parameters from a similar operating condition"""
        if not self.condition_parameters:
            return None
        
        best_similarity = 0.0
        best_params = None
        
        for existing_condition, params in self.condition_parameters.items():
            similarity = self._calculate_condition_similarity(condition, existing_condition)
            
            if similarity > best_similarity and similarity > self.config.similarity_threshold:
                best_similarity = similarity
                best_params = params
        
        return best_params
    
    def _calculate_condition_similarity(self, cond1: OperatingCondition, 
                                      cond2: OperatingCondition) -> float:
        """Calculate similarity between two operating conditions"""
        # Speed range similarity (normalized overlap)
        speed_overlap = max(0, min(cond1.speed_range[1], cond2.speed_range[1]) - 
                          max(cond1.speed_range[0], cond2.speed_range[0]))
        speed_union = max(cond1.speed_range[1], cond2.speed_range[1]) - \
                     min(cond1.speed_range[0], cond2.speed_range[0])
        speed_sim = speed_overlap / speed_union if speed_union > 0 else 0
        
        # Acceleration type similarity
        accel_sim = 1.0 if cond1.acceleration_type == cond2.acceleration_type else 0.3
        
        # Gradient level similarity
        gradient_sim = 1.0 if cond1.gradient_level == cond2.gradient_level else 0.5
        
        # Weighted average
        return 0.5 * speed_sim + 0.3 * accel_sim + 0.2 * gradient_sim
    
    def _calculate_contextual_score(self, perf: ContextualPerformance) -> float:
        """Calculate performance score for a specific condition"""
        # Weight factors (adjusted for dynamic conditions)
        w_tracking = 25.0     # Higher weight on tracking in dynamic conditions
        w_smoothness = 8.0    # Slightly lower weight on smoothness
        w_success = 20.0      # High weight on solver reliability
        
        # Normalize scores
        tracking_score = perf.tracking_rms
        if tracking_score > 1.0:  # Heavy penalty for >1kph error
            tracking_score *= 3.0
        
        smoothness_score = min(perf.control_smoothness, 100.0) / 100.0
        success_score = 1.0 - perf.success_rate
        
        if success_score > 0.1:  # Heavy penalty for low success rate
            success_score *= 5.0
        
        total_score = (w_tracking * tracking_score + 
                      w_smoothness * smoothness_score + 
                      w_success * success_score)
        
        return total_score
    
    def _get_condition_best_score(self, condition: OperatingCondition) -> float:
        """Get the best score achieved for a specific condition"""
        if condition not in self.condition_performance:
            return float('inf')
        
        perf = self.condition_performance[condition]
        perf.update_metrics()
        return self._calculate_contextual_score(perf)
    
    def _optimize_for_condition(self, condition: OperatingCondition, 
                               perf: ContextualPerformance) -> DeePCParameters:
        """
        Optimize parameters for a specific operating condition using gradient-free method
        """
        current_params = self.condition_parameters.get(condition, self.current_params)
        
        # Simple adaptive step based on condition characteristics
        if condition.acceleration_type == 'accel':
            # More aggressive tracking for acceleration
            Q_adjustment = 1.2
            R_adjustment = 0.9
        elif condition.acceleration_type == 'decel':
            # More conservative for deceleration
            Q_adjustment = 1.1
            R_adjustment = 1.1
        else:  # cruise
            # Balanced for cruising
            Q_adjustment = 1.0
            R_adjustment = 1.0
        
        # Speed-based adjustments
        avg_speed = np.mean(perf.reference_speeds[-20:])  # Recent average
        if avg_speed > 80:  # High speed
            lambda_adjustments = 1.2  # More regularization at high speed
        elif avg_speed < 25:  # Low speed
            lambda_adjustments = 0.9  # Less regularization at low speed
        else:
            lambda_adjustments = 1.0
        
        # Create optimized parameters
        new_params = DeePCParameters(
            Tini=self.fixed_Tini,
            THorizon=self.fixed_THorizon,
            hankel_subB_size=self.fixed_hankel_size,
            
            # Adjusted weights
            Q_val=np.clip(current_params.Q_val * Q_adjustment, 
                         self.config.Q_bounds[0], self.config.Q_bounds[1]),
            R_val=np.clip(current_params.R_val * R_adjustment,
                         self.config.R_bounds[0], self.config.R_bounds[1]),
            
            # Adjusted regularization
            lambda_g_val=np.clip(current_params.lambda_g_val * lambda_adjustments,
                                self.config.lambda_g_bounds[0], self.config.lambda_g_bounds[1]),
            lambda_y_val=np.clip(current_params.lambda_y_val * lambda_adjustments,
                                self.config.lambda_y_bounds[0], self.config.lambda_y_bounds[1]),
            lambda_u_val=np.clip(current_params.lambda_u_val * lambda_adjustments,
                                self.config.lambda_u_bounds[0], self.config.lambda_u_bounds[1]),
            
            # Preserve other parameters
            decay_rate_q=current_params.decay_rate_q,
            decay_rate_r=current_params.decay_rate_r,
            solver_type=current_params.solver_type
        )
        
        return new_params
    
    def should_update_parameters(self) -> bool:
        """Check if parameters should be updated for current condition"""
        if not self.is_active or self.current_condition is None:
            return False
        
        with self.lock:
            if self.current_condition not in self.condition_performance:
                return False
            
            perf = self.condition_performance[self.current_condition]
            return len(perf.tracking_errors) >= self.config.min_samples_per_condition
    
    def get_current_parameters(self) -> DeePCParameters:
        """Get parameters optimized for current operating condition"""
        with self.lock:
            return self.current_params
    
    def save_adaptive_results(self, filepath: str):
        """Save adaptive tuning results"""
        results = {
            'total_updates': self.total_updates,
            'global_best_score': self.global_best_score,
            'global_best_params': asdict(self.global_best_params),
            'condition_visits': dict(self.condition_visits),
            'condition_parameters': {
                str(cond): asdict(params) 
                for cond, params in self.condition_parameters.items()
            },
            'condition_performance': {
                str(cond): {
                    'tracking_rms': perf.tracking_rms,
                    'success_rate': perf.success_rate,
                    'sample_count': perf.sample_count
                } for cond, perf in self.condition_performance.items()
            },
            'config': asdict(self.config)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[AdaptiveTuner] Saved adaptive tuning results to {filepath}")
    
    def generate_adaptive_report(self) -> str:
        """Generate comprehensive adaptive tuning report"""
        report = f"""
Adaptive DeePC Parameter Tuning Report
======================================

Summary:
--------
Total parameter updates: {self.total_updates}
Unique conditions encountered: {len(self.condition_performance)}
Global best score: {self.global_best_score:.3f}

Operating Conditions Analysis:
------------------------------"""
        
        for condition, perf in self.condition_performance.items():
            perf.update_metrics()
            score = self._calculate_contextual_score(perf)
            visits = self.condition_visits[condition]
            
            report += f"""

Condition: {condition}
  Visits: {visits}
  Samples: {perf.sample_count}
  Tracking RMS: {perf.tracking_rms:.3f} kph
  Success Rate: {perf.success_rate:.1%}
  Score: {score:.3f}"""
        
        # Show parameter variations across conditions
        if len(self.condition_parameters) > 1:
            report += f"""

Parameter Variation Across Conditions:
-------------------------------------"""
            Q_values = [p.Q_val for p in self.condition_parameters.values()]
            R_values = [p.R_val for p in self.condition_parameters.values()]
            
            report += f"""
Q_val range: {min(Q_values):.1f} - {max(Q_values):.1f}
R_val range: {min(R_values):.4f} - {max(R_values):.4f}"""
        
        return report

# Integration helper
def create_adaptive_tuner(initial_params: DeePCParameters) -> AdaptiveOnlineTuner:
    """Create adaptive tuner with sensible defaults"""
    config = AdaptiveTuningConfig(
        speed_bins=[(0, 25), (25, 50), (50, 80), (80, 120)],
        min_samples_per_condition=30,     # Reduced for faster adaptation
        max_samples_per_condition=150,    # Reasonable memory usage
        similarity_threshold=0.6          # Allow reasonable parameter reuse
    )
    
    return AdaptiveOnlineTuner(config, initial_params)

if __name__ == "__main__":
    # Test the adaptive tuner
    from DeePCParameterTuner import DeePCParameters
    
    initial_params = DeePCParameters(
        Tini=20, THorizon=20, hankel_subB_size=80,
        Q_val=400.0, R_val=0.08, lambda_g_val=80.0,
        lambda_y_val=15.0, lambda_u_val=15.0,
        decay_rate_q=0.85, decay_rate_r=0.12
    )
    
    tuner = create_adaptive_tuner(initial_params)
    print("Adaptive online tuner created successfully!")
    print(f"Speed bins: {tuner.config.speed_bins}")