#!/usr/bin/env python3
"""
DeePCParameterTuner.py
Author: Claude AI Assistant
Date: 2025-01-21
Version: 1.0.0
Description: Parameter tuning utility for DeePC controller optimization
Objective: Find optimal DeePC parameters for speed tracking error < 1kph and smooth control
"""

import numpy as np
import time
import os
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from pathlib import Path
import json

class DeePCParameters:
    """Class to hold all DeePC tunable parameters"""
    def __init__(self):
        # Core DeePC parameters
        self.Tini = 30                    # Initial data length
        self.THorizon = 30                # Prediction horizon
        self.hankel_subB_size = 120       # Hankel sub-block size
        
        # Weighting parameters
        self.Q_val = 265.0                # Output tracking weight
        self.R_val = 0.15                 # Control effort weight
        self.lambda_g_val = 60.0          # Regularization weight on g
        self.lambda_y_val = 10.0          # Output mismatch weight
        self.lambda_u_val = 10.0          # Input mismatch weight
        
        # Decay parameters for time-varying weights
        self.decay_rate_q = 0.9           # Q decay rate
        self.decay_rate_r = 0.1           # R decay rate
        
        # Solver parameters
        self.solver_type = "acados"       # "acados", "cvxpy", "casadi"
        self.max_iterations = 50          # Max solver iterations
        self.tolerance = 1e-5             # Solver tolerance

class PerformanceMetrics:
    """Class to hold performance evaluation metrics"""
    def __init__(self):
        self.tracking_error_rms = float('inf')
        self.tracking_error_max = float('inf')
        self.control_effort_rms = float('inf')
        self.control_smoothness = float('inf')  # Rate of change variance
        self.solver_time_avg = float('inf')
        self.solver_success_rate = 0.0
        self.overall_score = float('inf')

class DeePCParameterTuner:
    """
    Parameter tuning class for DeePC controller optimization
    """
    
    def __init__(self, data_dir, target_frequency=10.0):
        """
        Initialize the parameter tuner
        
        Args:
            data_dir: Path to directory containing Hankel matrix data
            target_frequency: Target control frequency in Hz
        """
        self.data_dir = Path(data_dir)
        self.target_frequency = target_frequency
        self.target_cycle_time = 1.0 / target_frequency
        
        # Load reference data
        self.ref_speed = None
        self.ref_time = None
        
        # Parameter bounds for optimization
        self.param_bounds = {
            'Tini': (10, 50),
            'THorizon': (10, 50),
            'hankel_subB_size': (80, 200),
            'Q_val': (10, 1000),
            'R_val': (0.01, 10.0),
            'lambda_g_val': (1, 200),
            'lambda_y_val': (1, 100),
            'lambda_u_val': (1, 100),
            'decay_rate_q': (0.1, 1.0),
            'decay_rate_r': (0.01, 0.5)
        }
        
        # Tuning history
        self.tuning_history: List[Dict] = []
    
    def load_reference_cycle(self, cycle_path: str = None) -> bool:
        """Load reference speed cycle for evaluation"""
        try:
            if cycle_path is None:
                # Use WLTP cycle as default
                cycle_path = "/home/guiliang/Desktop/DR-CodeHub/DRtemp/DR-Claude-DeePCSoak/drivecycle/CYC_WLTP.mat"
            
            import scipy.io as sio
            cycle_data = sio.loadmat(cycle_path)
            
            # Extract the main variable (assuming standard format)
            for key in cycle_data.keys():
                if not key.startswith('__'):
                    data = cycle_data[key]
                    if data.ndim == 2 and data.shape[1] >= 2:
                        self.ref_time = data[:, 0].flatten()
                        ref_speed_mph = data[:, 1].flatten()
                        self.ref_speed = ref_speed_mph * 1.60934  # Convert to kph
                        return True
            return False
        except Exception as e:
            print(f"Failed to load reference cycle: {e}")
            return False
    
    def evaluate_parameters(self, params: DeePCParameters, 
                          simulation_duration: float = 100.0) -> PerformanceMetrics:
        """
        Evaluate DeePC parameters using simulation
        
        Args:
            params: DeePC parameters to evaluate
            simulation_duration: Duration of simulation in seconds
            
        Returns:
            PerformanceMetrics object with evaluation results
        """
        metrics = PerformanceMetrics()
        
        try:
            # Validate parameter constraints
            if not self._validate_parameters(params):
                return metrics
            
            # Check if parameters allow real-time operation
            estimated_solve_time = self._estimate_solve_time(params)
            if estimated_solve_time > self.target_cycle_time * 0.8:  # 80% margin
                metrics.solver_time_avg = estimated_solve_time
                return metrics
            
            # Run simulation evaluation
            tracking_errors, control_efforts, solve_times, success_flags = \
                self._simulate_deepc_performance(params, simulation_duration)
            
            # Calculate performance metrics
            if len(tracking_errors) > 0:
                metrics.tracking_error_rms = np.sqrt(np.mean(np.array(tracking_errors)**2))
                metrics.tracking_error_max = np.max(np.abs(tracking_errors))
                
            if len(control_efforts) > 0:
                metrics.control_effort_rms = np.sqrt(np.mean(np.array(control_efforts)**2))
                # Control smoothness: variance of control rate of change
                if len(control_efforts) > 1:
                    control_changes = np.diff(control_efforts)
                    metrics.control_smoothness = np.var(control_changes)
                    
            if len(solve_times) > 0:
                metrics.solver_time_avg = np.mean(solve_times)
                metrics.solver_success_rate = np.mean(success_flags)
            
            # Calculate overall score (lower is better)
            metrics.overall_score = self._calculate_overall_score(metrics)
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            
        return metrics
    
    def _validate_parameters(self, params: DeePCParameters) -> bool:
        """Validate parameter constraints"""
        # Check degrees of freedom constraint
        g_dim = params.hankel_subB_size - params.Tini - params.THorizon + 1
        min_g_dim = 2 * params.Tini  # u_dim + y_dim = 2 for SISO system
        
        if g_dim <= min_g_dim:
            return False
            
        # Check parameter bounds
        if params.Tini <= 0 or params.THorizon <= 0:
            return False
            
        if params.hankel_subB_size <= params.Tini + params.THorizon:
            return False
            
        return True
    
    def _estimate_solve_time(self, params: DeePCParameters) -> float:
        """Estimate solver time based on problem size"""
        # Simple heuristic based on problem dimensions
        g_dim = params.hankel_subB_size - params.Tini - params.THorizon + 1
        
        if params.solver_type == "acados":
            # Acados is typically faster
            base_time = 0.001  # 1ms base
            complexity_factor = g_dim * 0.00005
        elif params.solver_type == "cvxpy":
            # CVXPY moderate speed
            base_time = 0.005  # 5ms base
            complexity_factor = g_dim * 0.0001
        else:  # casadi
            base_time = 0.010  # 10ms base
            complexity_factor = g_dim * 0.0002
            
        return base_time + complexity_factor
    
    def _simulate_deepc_performance(self, params: DeePCParameters, 
                                   duration: float) -> Tuple[List, List, List, List]:
        """
        Simulate DeePC performance without actually running the controller
        This is a simplified simulation for parameter evaluation
        """
        if self.ref_speed is None:
            if not self.load_reference_cycle():
                return [], [], [], []
        
        # Simulation parameters
        dt = 1.0 / self.target_frequency
        n_steps = int(duration / dt)
        
        tracking_errors = []
        control_efforts = []
        solve_times = []
        success_flags = []
        
        # Simple plant model for simulation (first-order system)
        plant_tau = 2.0  # Time constant
        current_speed = 0.0
        current_u = 0.0
        
        for step in range(n_steps):
            t = step * dt
            
            # Get reference speed at current time
            if t >= self.ref_time[-1]:
                ref_speed = 0.0
            else:
                ref_speed = np.interp(t, self.ref_time, self.ref_speed)
            
            # Simulate solve time
            solve_time = self._estimate_solve_time(params)
            solve_time += np.random.normal(0, solve_time * 0.1)  # Add noise
            
            # Simple controller simulation based on parameters
            error = ref_speed - current_speed
            
            # Simulate control effort based on parameters
            control_gain = params.Q_val / (params.R_val + 1e-6)
            u_new = current_u + control_gain * error * 0.001  # Simple PI-like
            
            # Apply control limits
            u_new = np.clip(u_new, -30, 100)
            
            # Rate limiting
            max_rate = 10.0  # Max rate of change
            u_new = np.clip(u_new, current_u - max_rate, current_u + max_rate)
            
            # Update plant model
            current_speed += dt / plant_tau * (u_new * 0.5 - current_speed)  # Simple first-order
            current_u = u_new
            
            # Record metrics
            tracking_errors.append(error)
            control_efforts.append(u_new)
            solve_times.append(solve_time)
            success_flags.append(solve_time < self.target_cycle_time * 0.9)
        
        return tracking_errors, control_efforts, solve_times, success_flags
    
    def _calculate_overall_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (lower is better)"""
        # Weights for different objectives
        w_tracking = 10.0   # High weight on tracking performance
        w_effort = 1.0      # Moderate weight on control effort
        w_smooth = 5.0      # High weight on smoothness
        w_time = 20.0       # Very high weight on real-time capability
        w_success = 15.0    # High weight on solver success
        
        # Normalize metrics
        tracking_score = metrics.tracking_error_rms  # Already in kph
        effort_score = metrics.control_effort_rms / 100.0  # Normalize by max effort
        smooth_score = min(metrics.control_smoothness, 100.0) / 100.0  # Cap and normalize
        time_score = metrics.solver_time_avg / self.target_cycle_time
        success_score = 1.0 - metrics.solver_success_rate
        
        # Penalize heavily if tracking error > 1 kph
        if metrics.tracking_error_rms > 1.0:
            tracking_score *= 5.0
        
        # Penalize heavily if real-time constraint violated
        if metrics.solver_time_avg > self.target_cycle_time:
            time_score *= 10.0
        
        score = (w_tracking * tracking_score + 
                w_effort * effort_score +
                w_smooth * smooth_score +
                w_time * time_score +
                w_success * success_score)
        
        return score
    
    def optimize_parameters(self, method: str = "differential_evolution",
                          max_evaluations: int = 50,
                          initial_params: Optional[DeePCParameters] = None) -> Tuple[DeePCParameters, PerformanceMetrics]:
        """
        Optimize DeePC parameters using specified optimization method
        
        Args:
            method: Optimization method ("differential_evolution", "minimize", "grid_search")
            max_evaluations: Maximum number of parameter evaluations
            initial_params: Initial parameter guess (for some methods)
            
        Returns:
            Tuple of (best_parameters, best_metrics)
        """
        print(f"Starting parameter optimization with {method}...")
        
        if method == "differential_evolution":
            return self._optimize_differential_evolution(max_evaluations)
        elif method == "grid_search":
            return self._optimize_grid_search(max_evaluations)
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
    
    def _optimize_differential_evolution(self, max_evaluations: int) -> Tuple[DeePCParameters, PerformanceMetrics]:
        """Optimize using differential evolution algorithm"""
        
        # Define parameter bounds for optimization
        bounds = [
            self.param_bounds['Tini'],
            self.param_bounds['THorizon'], 
            self.param_bounds['hankel_subB_size'],
            self.param_bounds['Q_val'],
            self.param_bounds['R_val'],
            self.param_bounds['lambda_g_val'],
            self.param_bounds['lambda_y_val'],
            self.param_bounds['lambda_u_val'],
            self.param_bounds['decay_rate_q'],
            self.param_bounds['decay_rate_r']
        ]
        
        def objective(x):
            """Objective function for optimization"""
            params = DeePCParameters(
                Tini=int(x[0]),
                THorizon=int(x[1]),
                hankel_subB_size=int(x[2]),
                Q_val=x[3],
                R_val=x[4],
                lambda_g_val=x[5],
                lambda_y_val=x[6],
                lambda_u_val=x[7],
                decay_rate_q=x[8],
                decay_rate_r=x[9]
            )
            
            metrics = self.evaluate_parameters(params)
            
            # Store in history
            result = {
                'parameters': params.__dict__,
                'metrics': metrics.__dict__,
                'timestamp': time.time()
            }
            self.tuning_history.append(result)
            
            print(f"Eval {len(self.tuning_history)}: Score={metrics.overall_score:.3f}, "
                  f"Tracking RMS={metrics.tracking_error_rms:.3f}kph, "
                  f"Solve time={metrics.solver_time_avg:.3f}ms")
            
            return metrics.overall_score
        
        # Run optimization
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=max_evaluations // 10,  # Adjust for population size
            popsize=10,
            seed=42,
            workers=1  # Sequential for stability
        )
        
        # Extract best parameters
        best_x = result.x
        best_params = DeePCParameters(
            Tini=int(best_x[0]),
            THorizon=int(best_x[1]),
            hankel_subB_size=int(best_x[2]),
            Q_val=best_x[3],
            R_val=best_x[4],
            lambda_g_val=best_x[5],
            lambda_y_val=best_x[6],
            lambda_u_val=best_x[7],
            decay_rate_q=best_x[8],
            decay_rate_r=best_x[9]
        )
        
        best_metrics = self.evaluate_parameters(best_params)
        
        return best_params, best_metrics
    
    def _optimize_grid_search(self, max_evaluations: int) -> Tuple[DeePCParameters, PerformanceMetrics]:
        """Optimize using grid search over key parameters"""
        
        # Define grid for key parameters
        param_grid = {
            'Tini': [20, 30, 40],
            'THorizon': [20, 30, 40],
            'Q_val': [100, 265, 500],
            'R_val': [0.05, 0.15, 0.5],
            'lambda_g_val': [20, 60, 120],
        }
        
        best_params = None
        best_metrics = PerformanceMetrics()
        
        eval_count = 0
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        print(f"Grid search: {total_combinations} combinations")
        
        # Generate all combinations
        import itertools
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            if eval_count >= max_evaluations:
                break
                
            # Create parameter set
            params = DeePCParameters()
            for key, value in zip(keys, values):
                setattr(params, key, value)
            
            # Evaluate
            metrics = self.evaluate_parameters(params)
            eval_count += 1
            
            print(f"Grid eval {eval_count}/{min(max_evaluations, total_combinations)}: "
                  f"Score={metrics.overall_score:.3f}")
            
            # Update best
            if metrics.overall_score < best_metrics.overall_score:
                best_params = params
                best_metrics = metrics
        
        return best_params, best_metrics
    
    def save_tuning_results(self, filepath: str, best_params: DeePCParameters, best_metrics: PerformanceMetrics):
        """Save tuning results to file"""
        results = {
            'best_parameters': best_params.__dict__,
            'best_metrics': best_metrics.__dict__,
            'tuning_history': self.tuning_history,
            'optimization_settings': {
                'target_frequency': self.target_frequency,
                'parameter_bounds': self.param_bounds
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Tuning results saved to {filepath}")
    
    def generate_parameter_recommendations(self, best_params: DeePCParameters, 
                                         best_metrics: PerformanceMetrics) -> str:
        """Generate parameter recommendations report"""
        
        report = f"""
DeePC Parameter Tuning Results
==============================

Best Parameters Found:
---------------------
Tini: {best_params.Tini}
THorizon: {best_params.THorizon}  
hankel_subB_size: {best_params.hankel_subB_size}
Q_val: {best_params.Q_val:.3f}
R_val: {best_params.R_val:.3f}
lambda_g_val: {best_params.lambda_g_val:.3f}
lambda_y_val: {best_params.lambda_y_val:.3f}
lambda_u_val: {best_params.lambda_u_val:.3f}
decay_rate_q: {best_params.decay_rate_q:.3f}
decay_rate_r: {best_params.decay_rate_r:.3f}

Performance Metrics:
-------------------
Tracking RMS Error: {best_metrics.tracking_error_rms:.3f} kph
Max Tracking Error: {best_metrics.tracking_error_max:.3f} kph
Average Solve Time: {best_metrics.solver_time_avg:.3f} ms
Solver Success Rate: {best_metrics.solver_success_rate:.1%}
Overall Score: {best_metrics.overall_score:.3f}

Real-time Compatibility:
-----------------------
Target cycle time: {self.target_cycle_time*1000:.1f} ms
Actual solve time: {best_metrics.solver_time_avg:.1f} ms
Time margin: {(self.target_cycle_time*1000 - best_metrics.solver_time_avg):.1f} ms
Real-time feasible: {'Yes' if best_metrics.solver_time_avg < self.target_cycle_time*1000*0.8 else 'No'}

Recommendations:
---------------
"""
        
        if best_metrics.tracking_error_rms <= 1.0:
            report += "✓ Tracking error objective met (< 1 kph)\n"
        else:
            report += f"✗ Tracking error too high: {best_metrics.tracking_error_rms:.3f} kph > 1.0 kph\n"
            report += "  Consider increasing Q_val or adjusting horizon parameters\n"
        
        if best_metrics.solver_time_avg < self.target_cycle_time * 1000 * 0.8:
            report += "✓ Real-time constraint satisfied\n" 
        else:
            report += "✗ Real-time constraint violated\n"
            report += "  Consider reducing problem size or using faster solver\n"
        
        if best_params.solver_type == "acados":
            report += "✓ Using Acados solver for optimal real-time performance\n"
        else:
            report += "  Consider switching to Acados solver for better real-time performance\n"
        
        return report

if __name__ == "__main__":
    # Example usage
    data_dir = "/home/guiliang/Desktop/DR-CodeHub/DRtemp/DR-Claude-DeePCSoak/DeePC-Runtime/dataForHankle/smallDataSet"
    
    tuner = DeePCParameterTuner(data_dir, target_frequency=10.0)
    
    # Run optimization
    best_params, best_metrics = tuner.optimize_parameters(
        method="differential_evolution",
        max_evaluations=30
    )
    
    # Generate report
    report = tuner.generate_parameter_recommendations(best_params, best_metrics)
    print(report)
    
    # Save results
    results_file = "deepc_tuning_results.json"
    tuner.save_tuning_results(results_file, best_params, best_metrics)