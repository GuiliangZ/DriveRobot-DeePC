#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deepc_config.py
Configuration file for DeePC controller settings
Modify these parameters to tune the controller behavior
"""

from DeePCParametersSimple import DeePCParameters

# --- SOLVER CONFIGURATION ---
SOLVER_TYPE = "acados"                   # Options: "acados", "cvxpy"
CONTROL_FREQUENCY = 10.0                  # Control frequency in Hz (10Hz for moderate real-time performance)

# --- PARAMETER TUNING OPTIONS ---
ENABLE_PARAMETER_TUNING = False          # Set to True to run parameter optimization before control
TUNING_METHOD = "differential_evolution"  # Options: "differential_evolution", "grid_search"  
MAX_TUNING_EVALUATIONS = 30              # Number of parameter combinations to evaluate
 
# --- ONLINE PARAMETER TUNING OPTIONS ---
ENABLE_ONLINE_TUNING = True              # Enable hardware-in-the-loop parameter tuning
ONLINE_TUNING_MODE = "adaptive"          # Options: "basic", "adaptive" 
ONLINE_TUNING_START_DELAY = 100          # Control cycles to wait before starting online tuning
EVALUATION_WINDOW_SIZE = 100             # Control cycles per parameter evaluation (10s at 10Hz) - for basic mode
PARAMETER_UPDATE_INTERVAL = 50           # Cycles between parameter updates during evaluation - for basic mode

# --- ADAPTIVE ONLINE TUNING OPTIONS ---
# These options are used when ONLINE_TUNING_MODE = "adaptive"
ADAPTIVE_SPEED_BINS = [(0, 25), (25, 50), (50, 80), (80, 120)]  # Speed ranges for condition classification
ADAPTIVE_MIN_SAMPLES = 30                # Minimum samples per condition before tuning
ADAPTIVE_MAX_SAMPLES = 150               # Maximum samples to keep per condition  
ADAPTIVE_SIMILARITY_THRESHOLD = 0.6      # Threshold for reusing parameters from similar conditions
ADAPTIVE_ACCEL_THRESHOLD = 0.5           # kph/s threshold for acceleration/deceleration classification

# --- MANUAL DeePC PARAMETERS ---
# These parameters are used if ENABLE_PARAMETER_TUNING = False
# If tuning is enabled, these serve as initial guesses

# Recommended parameters for 10Hz operation (optimized for speed tracking < 1kph error)
MANUAL_DEEPC_PARAMS = DeePCParameters(
    # Core DeePC structure parameters
    Tini=20,                              # Initial data length (2.5s at 10Hz)
    THorizon=20,                          # Prediction horizon (2.5s at 10Hz) 
    hankel_subB_size=80,                 # Hankel matrix size (10s of data at 10Hz)
    
    # Cost function weights (tuned for tracking vs smoothness)
    Q_val=400.0,                          # Output tracking weight (higher = better tracking)
    R_val=0.08,                           # Control effort weight (lower = more aggressive)
    
    # Regularization weights (tuned for robustness)
    lambda_g_val=80.0,                    # g-vector regularization (moderate)
    lambda_y_val=15.0,                    # Output mismatch tolerance (moderate)
    lambda_u_val=15.0,                    # Input mismatch tolerance (moderate)
    
    # Time-varying weights for better transient response
    decay_rate_q=0.85,                    # Q decay rate (focus on near-term tracking)
    decay_rate_r=0.12,                    # R decay rate (moderate control smoothing)
    
    # Solver settings
    solver_type="acados",                 # Use acados for real-time performance
    max_iterations=50,                    # Solver iteration limit
    tolerance=1e-5                        # Solver tolerance
)

# Alternative parameter sets for different scenarios
# Uncomment and modify MANUAL_DEEPC_PARAMS assignment to use these

# More aggressive tracking (may be less smooth)
AGGRESSIVE_PARAMS = DeePCParameters(
    Tini=20, THorizon=20, hankel_subB_size=80,
    Q_val=600.0, R_val=0.05,
    lambda_g_val=40.0, lambda_y_val=8.0, lambda_u_val=8.0,
    decay_rate_q=0.9, decay_rate_r=0.08,
    solver_type="acados", max_iterations=50, tolerance=1e-5
)

# More conservative/smooth (may have slightly higher tracking error)
CONSERVATIVE_PARAMS = DeePCParameters(
    Tini=35, THorizon=35, hankel_subB_size=140,
    Q_val=200.0, R_val=0.2,
    lambda_g_val=120.0, lambda_y_val=25.0, lambda_u_val=25.0,
    decay_rate_q=0.7, decay_rate_r=0.15,
    solver_type="acados", max_iterations=50, tolerance=1e-5
)

# --- REAL-TIME PERFORMANCE SETTINGS ---
# Maximum allowed solve time as percentage of control period
MAX_SOLVE_TIME_RATIO = 0.8               # 80% of control period (60ms for 10Hz)

# Fallback to PID if DeePC solve time exceeds this limit
ENABLE_ADAPTIVE_FALLBACK = True

# --- DATA CACHING SETTINGS ---
USE_CACHED_HANKEL_DATA = True            # Reuse cached Hankel matrices
RECOMPILE_SOLVER = True                  # Set to True if changing constraint structure or parameters
                                         # NOTE: Always True for safety to avoid dimension mismatches

# --- CONSTRAINT SETTINGS ---
# Control input bounds (PWM percentage)
U_MIN = -30.0                            # Minimum control input (%)
U_MAX = 100.0                            # Maximum control input (%)

# Output bounds (vehicle speed)  
Y_MIN = 0.0                              # Minimum speed (kph)
Y_MAX = 140.0                            # Maximum speed (kph)

# Rate limiting (for smoother control)
ENABLE_RATE_LIMITING = True
MAX_CONTROL_RATE = 80.0                  # Max PWM change per second

# --- LOGGING AND MONITORING ---
LOG_SOLVER_PERFORMANCE = True           # Log solve times and success rates
PRINT_PERFORMANCE_STATS = True          # Print performance stats every N cycles
PERFORMANCE_STATS_INTERVAL = 100        # Print stats every 100 control cycles

# --- PID CONTROLLER SETTINGS (Fallback Controller) ---
# PID parameters for fallback control when DeePC fails
PID_KP = 2.5                            # Proportional gain
PID_KI = 0.15                           # Integral gain  
PID_KD = 0.05                           # Derivative gain
PID_INTEGRAL_LIMIT = 50.0               # Anti-windup limit for integral term

# --- SAFETY SETTINGS ---
EMERGENCY_SPEED_LIMIT = 140.0           # Emergency brake if speed exceeds this
MIN_SOC_STOP = 2.2                      # Stop test if battery SOC drops below this

# --- CONFIGURATION VALIDATION ---
def validate_config():
    """Validate configuration parameters for consistency"""
    errors = []
    
    # Check control frequency
    if CONTROL_FREQUENCY <= 0 or CONTROL_FREQUENCY > 100:
        errors.append("CONTROL_FREQUENCY must be between 0 and 100 Hz")
    
    # Check DeePC parameter consistency
    params = MANUAL_DEEPC_PARAMS
    g_dim = params.hankel_subB_size - params.Tini - params.THorizon + 1
    min_g_dim = params.Tini + params.THorizon  # For SISO system
    
    if g_dim <= min_g_dim:
        errors.append("Insufficient degrees of freedom: g_dim={} <= {}".format(g_dim, min_g_dim))
    
    # Check solver compatibility
    if params.solver_type not in ["acados", "cvxpy"]:
        errors.append("solver_type must be 'acados' or 'cvxpy'")
    
    # Check bounds
    if U_MIN >= U_MAX:
        errors.append("U_MIN must be less than U_MAX")
    
    if Y_MIN >= Y_MAX:
        errors.append("Y_MIN must be less than Y_MAX")
    
    # Check time constraints
    control_period_ms = 1000.0 / CONTROL_FREQUENCY
    max_solve_time_ms = control_period_ms * MAX_SOLVE_TIME_RATIO
    
    if max_solve_time_ms < 5.0:  # Minimum 5ms for any reasonable solver
        errors.append("Max solve time too restrictive: {:.1f}ms".format(max_solve_time_ms))
    
    # Check PID parameters
    if PID_KP <= 0:
        errors.append("PID_KP must be positive")
    if PID_KI < 0:
        errors.append("PID_KI must be non-negative") 
    if PID_KD < 0:
        errors.append("PID_KD must be non-negative")
    if PID_INTEGRAL_LIMIT <= 0:
        errors.append("PID_INTEGRAL_LIMIT must be positive")
    
    return errors

if __name__ == "__main__":
    # Validate configuration
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print("  - {}".format(error))
    else:
        print("Configuration validated successfully!")
        
        # Print summary
        params = MANUAL_DEEPC_PARAMS
        print("\nDeePC Configuration Summary:")
        print("  Control frequency: {} Hz".format(CONTROL_FREQUENCY))
        print("  Solver: {}".format(SOLVER_TYPE))
        print("  Parameter tuning: {}".format('Enabled' if ENABLE_PARAMETER_TUNING else 'Disabled'))
        print("  Tini: {}, THorizon: {}".format(params.Tini, params.THorizon))
        print("  Hankel size: {}".format(params.hankel_subB_size))
        print("  Q: {}, R: {}".format(params.Q_val, params.R_val))
        
        g_dim = params.hankel_subB_size - params.Tini - params.THorizon + 1
        print("  Decision variables (g_dim): {}".format(g_dim))
        
        control_period = 1000.0 / CONTROL_FREQUENCY
        max_solve_time = control_period * MAX_SOLVE_TIME_RATIO
        print("  Max solve time: {:.1f}ms per cycle".format(max_solve_time))
        
        print("\nPID Fallback Controller:")
        print("  Kp: {}, Ki: {}, Kd: {}".format(PID_KP, PID_KI, PID_KD))
        print("  Integral limit: {}".format(PID_INTEGRAL_LIMIT))