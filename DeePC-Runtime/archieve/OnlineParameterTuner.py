#!/usr/bin/env python3
"""
OnlineParameterTuner.py
Real-time hardware-in-the-loop parameter tuning for DeePC controller
Author: Claude AI Assistant
Date: 2025-01-21

This module enables online parameter tuning while the controller is running,
using actual vehicle response data to optimize DeePC parameters.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import threading
from scipy.optimize import minimize
from DeePCParameterTuner import DeePCParameters

@dataclass
class OnlineTuningConfig:
    """Configuration for online parameter tuning"""
    # Tuning intervals
    evaluation_window: int = 200        # Number of control cycles to evaluate each parameter set
    tuning_interval: int = 50           # Control cycles between parameter updates
    
    # Parameter bounds (only tunable parameters, keep structure fixed)
    Q_bounds: Tuple[float, float] = (50.0, 800.0)
    R_bounds: Tuple[float, float] = (0.01, 1.0)
    lambda_g_bounds: Tuple[float, float] = (10.0, 200.0)
    lambda_y_bounds: Tuple[float, float] = (1.0, 50.0)
    lambda_u_bounds: Tuple[float, float] = (1.0, 50.0)
    decay_q_bounds: Tuple[float, float] = (0.5, 0.99)
    decay_r_bounds: Tuple[float, float] = (0.01, 0.5)
    
    # Performance thresholds
    max_tracking_error: float = 1.0     # Maximum acceptable RMS tracking error (kph)
    max_control_effort_change: float = 20.0  # Maximum control effort change per second (%)
    min_solver_success_rate: float = 0.9  # Minimum solver success rate
    
    # Optimization settings
    step_size: float = 0.1              # Parameter update step size
    exploration_noise: float = 0.05     # Random exploration noise
    momentum: float = 0.3               # Momentum for parameter updates

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics from hardware"""
    tracking_errors: List[float]
    control_efforts: List[float] 
    solve_times: List[float]
    success_flags: List[bool]
    solver_costs: List[float]
    
    # Computed metrics
    tracking_rms: float = 0.0
    control_smoothness: float = 0.0  # Variance of control changes
    avg_solve_time: float = 0.0
    success_rate: float = 0.0
    avg_cost: float = 0.0
    
    def compute_metrics(self):
        """Compute derived metrics from raw data"""
        if len(self.tracking_errors) > 0:
            self.tracking_rms = np.sqrt(np.mean(np.array(self.tracking_errors)**2))
            
        if len(self.control_efforts) > 1:
            control_changes = np.diff(self.control_efforts)
            self.control_smoothness = np.var(control_changes)
            
        if len(self.solve_times) > 0:
            self.avg_solve_time = np.mean(self.solve_times)
            
        if len(self.success_flags) > 0:
            self.success_rate = np.mean(self.success_flags)
            
        if len(self.solver_costs) > 0:
            self.avg_cost = np.mean([c for c in self.solver_costs if c < float('inf')])

class OnlineParameterTuner:
    """
    Online parameter tuning system that runs during real-time control
    """
    
    def __init__(self, config: OnlineTuningConfig, initial_params: DeePCParameters):
        self.config = config
        self.current_params = initial_params
        
        # Fixed structure parameters (never change to avoid recompilation)
        self.fixed_Tini = initial_params.Tini
        self.fixed_THorizon = initial_params.THorizon  
        self.fixed_hankel_size = initial_params.hankel_subB_size
        
        # Tuning state
        self.is_tuning = False
        self.current_evaluation_cycle = 0
        self.total_evaluations = 0
        
        # Performance tracking
        self.current_metrics = PerformanceMetrics([], [], [], [], [])
        self.best_params = initial_params
        self.best_score = float('inf')
        self.tuning_history = []
        
        # Parameter momentum for smooth updates
        self.param_momentum = {
            'Q_val': 0.0,
            'R_val': 0.0, 
            'lambda_g_val': 0.0,
            'lambda_y_val': 0.0,
            'lambda_u_val': 0.0,
            'decay_rate_q': 0.0,
            'decay_rate_r': 0.0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        print(f"[OnlineTuner] Initialized with fixed structure:")
        print(f"  Tini: {self.fixed_Tini}, THorizon: {self.fixed_THorizon}")
        print(f"  Hankel size: {self.fixed_hankel_size}")
        print(f"  Evaluation window: {config.evaluation_window} cycles")
    
    def start_tuning(self):
        """Start online parameter tuning"""
        with self.lock:
            self.is_tuning = True
            self.current_evaluation_cycle = 0
            self.current_metrics = PerformanceMetrics([], [], [], [], [])
        print("[OnlineTuner] Started online parameter tuning")
    
    def stop_tuning(self):
        """Stop online parameter tuning"""
        with self.lock:
            self.is_tuning = False
        print("[OnlineTuner] Stopped online parameter tuning")
        
    def record_performance(self, tracking_error: float, control_effort: float, 
                         solve_time: float, success: bool, cost: float):
        """Record performance data from current control cycle"""
        if not self.is_tuning:
            return
            
        with self.lock:
            self.current_metrics.tracking_errors.append(abs(tracking_error))
            self.current_metrics.control_efforts.append(control_effort)
            self.current_metrics.solve_times.append(solve_time)
            self.current_metrics.success_flags.append(success)
            self.current_metrics.solver_costs.append(cost if cost != float('inf') else 1000.0)
            
            self.current_evaluation_cycle += 1
    
    def should_update_parameters(self) -> bool:
        """Check if it's time to update parameters"""
        with self.lock:
            return (self.is_tuning and 
                   self.current_evaluation_cycle >= self.config.evaluation_window)
    
    def update_parameters(self) -> Tuple[DeePCParameters, Dict]:
        """
        Update parameters based on current performance and return new parameters
        """
        if not self.should_update_parameters():
            return self.current_params, {}
            
        with self.lock:
            # Compute performance metrics
            self.current_metrics.compute_metrics()
            
            # Calculate performance score (lower is better)
            score = self._calculate_performance_score(self.current_metrics)
            
            # Update best parameters if improved
            if score < self.best_score:
                self.best_score = score
                self.best_params = DeePCParameters(
                    Tini=self.fixed_Tini,
                    THorizon=self.fixed_THorizon,
                    hankel_subB_size=self.fixed_hankel_size,
                    Q_val=self.current_params.Q_val,
                    R_val=self.current_params.R_val,
                    lambda_g_val=self.current_params.lambda_g_val,
                    lambda_y_val=self.current_params.lambda_y_val,
                    lambda_u_val=self.current_params.lambda_u_val,
                    decay_rate_q=self.current_params.decay_rate_q,
                    decay_rate_r=self.current_params.decay_rate_r,
                    solver_type=self.current_params.solver_type
                )
                print(f"[OnlineTuner] New best parameters found! Score: {score:.3f}")
            
            # Save tuning history
            self.tuning_history.append({
                'evaluation': self.total_evaluations,
                'params': self.current_params.__dict__.copy(),
                'metrics': {
                    'tracking_rms': self.current_metrics.tracking_rms,
                    'control_smoothness': self.current_metrics.control_smoothness,
                    'avg_solve_time': self.current_metrics.avg_solve_time,
                    'success_rate': self.current_metrics.success_rate,
                    'score': score
                },
                'timestamp': time.time()
            })
            
            # Generate next parameter set
            new_params = self._generate_next_parameters(score)
            
            # Reset evaluation metrics
            self.current_metrics = PerformanceMetrics([], [], [], [], [])
            self.current_evaluation_cycle = 0
            self.total_evaluations += 1
            
            # Print progress
            print(f"[OnlineTuner] Evaluation {self.total_evaluations} completed:")
            print(f"  Tracking RMS: {self.current_metrics.tracking_rms:.3f} kph")
            print(f"  Success rate: {self.current_metrics.success_rate:.1%}")
            print(f"  Score: {score:.3f} (Best: {self.best_score:.3f})")
            
            return new_params, self.tuning_history[-1]['metrics']
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (lower is better)"""
        # Weights for different objectives
        w_tracking = 20.0    # Very high weight on tracking performance  
        w_smoothness = 10.0  # High weight on control smoothness
        w_solve_time = 5.0   # Medium weight on solve time
        w_success = 15.0     # High weight on solver reliability
        
        # Normalize and penalize
        tracking_score = metrics.tracking_rms
        if tracking_score > self.config.max_tracking_error:
            tracking_score *= 3.0  # Heavy penalty for exceeding 1kph
            
        smoothness_score = min(metrics.control_smoothness, 100.0) / 100.0
        
        time_score = metrics.avg_solve_time / 100.0  # Normalize to ~100ms
        if time_score > 1.0:  # If solve time > 100ms
            time_score *= 2.0
            
        success_score = 1.0 - metrics.success_rate
        if success_score > (1.0 - self.config.min_solver_success_rate):
            success_score *= 5.0  # Heavy penalty for low success rate
        
        total_score = (w_tracking * tracking_score + 
                      w_smoothness * smoothness_score +
                      w_solve_time * time_score + 
                      w_success * success_score)
        
        return total_score
    
    def _generate_next_parameters(self, current_score: float) -> DeePCParameters:
        """Generate next parameter set using gradient-free optimization"""
        
        # Simple hill-climbing with momentum and exploration
        param_names = ['Q_val', 'R_val', 'lambda_g_val', 'lambda_y_val', 
                      'lambda_u_val', 'decay_rate_q', 'decay_rate_r']
        
        new_params = DeePCParameters(
            Tini=self.fixed_Tini,
            THorizon=self.fixed_THorizon, 
            hankel_subB_size=self.fixed_hankel_size,
            solver_type=self.current_params.solver_type
        )
        
        # Parameter bounds mapping
        bounds_map = {
            'Q_val': self.config.Q_bounds,
            'R_val': self.config.R_bounds,
            'lambda_g_val': self.config.lambda_g_bounds,
            'lambda_y_val': self.config.lambda_y_bounds,
            'lambda_u_val': self.config.lambda_u_bounds,
            'decay_rate_q': self.config.decay_q_bounds,
            'decay_rate_r': self.config.decay_r_bounds
        }
        
        for param_name in param_names:
            current_val = getattr(self.current_params, param_name)
            bounds = bounds_map[param_name]
            
            # Add exploration noise
            noise = np.random.normal(0, self.config.exploration_noise) * (bounds[1] - bounds[0])
            
            # Simple gradient estimation (if we're worse than best, try moving towards best)
            if current_score > self.best_score * 1.05:  # 5% tolerance
                best_val = getattr(self.best_params, param_name)
                gradient_direction = (best_val - current_val) * self.config.step_size
            else:
                # Random exploration if we're doing well
                gradient_direction = noise * 2
            
            # Apply momentum
            momentum_term = self.config.momentum * self.param_momentum[param_name]
            self.param_momentum[param_name] = gradient_direction + momentum_term
            
            # Update parameter
            new_val = current_val + self.param_momentum[param_name] + noise
            
            # Apply bounds
            new_val = np.clip(new_val, bounds[0], bounds[1])
            
            setattr(new_params, param_name, new_val)
        
        self.current_params = new_params
        return new_params
    
    def get_current_best(self) -> Tuple[DeePCParameters, float]:
        """Get current best parameters and score"""
        with self.lock:
            return self.best_params, self.best_score
    
    def save_tuning_results(self, filepath: str):
        """Save tuning results to file"""
        results = {
            'best_parameters': self.best_params.__dict__,
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            'tuning_history': self.tuning_history,
            'config': self.config.__dict__,
            'fixed_structure': {
                'Tini': self.fixed_Tini,
                'THorizon': self.fixed_THorizon,
                'hankel_subB_size': self.fixed_hankel_size
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[OnlineTuner] Saved tuning results to {filepath}")
    
    def generate_tuning_report(self) -> str:
        """Generate a summary report of the tuning process"""
        if len(self.tuning_history) == 0:
            return "No tuning data available."
        
        # Calculate improvement
        initial_score = self.tuning_history[0]['metrics']['score']
        improvement = ((initial_score - self.best_score) / initial_score) * 100
        
        report = f"""
Online DeePC Parameter Tuning Report
====================================

Tuning Summary:
--------------
Total Evaluations: {self.total_evaluations}
Initial Score: {initial_score:.3f}
Best Score: {self.best_score:.3f}
Improvement: {improvement:.1f}%

Fixed Structure Parameters:
--------------------------
Tini: {self.fixed_Tini}
THorizon: {self.fixed_THorizon}
Hankel Size: {self.fixed_hankel_size}

Best Parameters Found:
---------------------
Q_val: {self.best_params.Q_val:.2f}
R_val: {self.best_params.R_val:.4f}
lambda_g_val: {self.best_params.lambda_g_val:.2f}
lambda_y_val: {self.best_params.lambda_y_val:.2f}
lambda_u_val: {self.best_params.lambda_u_val:.2f}
decay_rate_q: {self.best_params.decay_rate_q:.3f}
decay_rate_r: {self.best_params.decay_rate_r:.3f}

Performance Trends:
------------------
"""
        
        if len(self.tuning_history) >= 3:
            # Show trend over last few evaluations
            recent = self.tuning_history[-3:]
            for i, entry in enumerate(recent):
                metrics = entry['metrics']
                report += f"Evaluation {entry['evaluation']}: "
                report += f"Tracking={metrics['tracking_rms']:.3f}kph, "
                report += f"Success={metrics['success_rate']:.1%}, "
                report += f"Score={metrics['score']:.3f}\n"
        
        return report

# Integration helper functions for main controller
def create_online_tuner(initial_params: DeePCParameters) -> OnlineParameterTuner:
    """Create and configure online parameter tuner"""
    config = OnlineTuningConfig(
        evaluation_window=100,  # 10 seconds at 10Hz
        tuning_interval=50,     # Update every 5 seconds  
        Q_bounds=(100.0, 600.0),
        R_bounds=(0.02, 0.5),
        lambda_g_bounds=(20.0, 150.0),
        lambda_y_bounds=(5.0, 30.0),
        lambda_u_bounds=(5.0, 30.0),
        decay_q_bounds=(0.6, 0.95),
        decay_r_bounds=(0.05, 0.3)
    )
    
    return OnlineParameterTuner(config, initial_params)

if __name__ == "__main__":
    # Example usage and testing
    from DeePCParameterTuner import DeePCParameters
    
    initial_params = DeePCParameters(
        Tini=25, THorizon=25, hankel_subB_size=100,
        Q_val=400.0, R_val=0.08, lambda_g_val=80.0,
        lambda_y_val=15.0, lambda_u_val=15.0,
        decay_rate_q=0.85, decay_rate_r=0.12
    )
    
    tuner = create_online_tuner(initial_params)
    print("Online parameter tuner created successfully!")
    print(f"Fixed structure preserved: Tini={tuner.fixed_Tini}, THorizon={tuner.fixed_THorizon}")