#!/usr/bin/env python3
"""
test_adaptive_tuning.py
Simple test script to validate the adaptive online tuning system
"""

import numpy as np
import time
from DeePCParameterTuner import DeePCParameters
from AdaptiveOnlineTuner import create_adaptive_tuner

def simulate_wltc_profile(duration_seconds=60, dt=0.1):
    """Simulate a WLTC-like speed profile with various driving conditions"""
    t = np.arange(0, duration_seconds, dt)
    
    # Create varied speed profile with different phases
    speed_profile = []
    ref_profile = []
    
    for i, time_val in enumerate(t):
        if time_val < 10:
            # Low speed urban
            ref_speed = 20 + 15 * np.sin(0.1 * time_val)
            measured_speed = ref_speed + np.random.normal(0, 0.5)
        elif time_val < 25:
            # Acceleration phase
            ref_speed = 20 + 2 * (time_val - 10)
            measured_speed = ref_speed + np.random.normal(0, 1.0)
        elif time_val < 35:
            # High speed cruise
            ref_speed = 80 + 10 * np.sin(0.2 * time_val)
            measured_speed = ref_speed + np.random.normal(0, 1.5)
        elif time_val < 45:
            # Deceleration
            ref_speed = 90 - 3 * (time_val - 35)
            measured_speed = ref_speed + np.random.normal(0, 1.0)
        else:
            # Mixed urban/suburban
            ref_speed = 40 + 20 * np.sin(0.15 * time_val) + 10 * np.cos(0.3 * time_val)
            measured_speed = ref_speed + np.random.normal(0, 0.8)
        
        # Keep speeds positive and reasonable
        ref_speed = max(5, min(120, ref_speed))
        measured_speed = max(0, min(125, measured_speed))
        
        speed_profile.append(measured_speed)
        ref_profile.append(ref_speed)
    
    return t, np.array(speed_profile), np.array(ref_profile)

def test_adaptive_tuning():
    """Test the adaptive online tuning system"""
    print("Testing Adaptive Online Parameter Tuning System")
    print("=" * 50)
    
    # Create initial parameters
    initial_params = DeePCParameters(
        Tini=20, THorizon=20, hankel_subB_size=80,
        Q_val=400.0, R_val=0.08, lambda_g_val=80.0,
        lambda_y_val=15.0, lambda_u_val=15.0,
        decay_rate_q=0.85, decay_rate_r=0.12
    )
    
    # Create adaptive tuner
    tuner = create_adaptive_tuner(initial_params)
    tuner.start_tuning()
    
    # Generate test data
    t, measured_speeds, ref_speeds = simulate_wltc_profile(duration_seconds=60)
    
    print(f"Simulating {len(t)} time steps ({len(t)/10:.1f} seconds)")
    print("Speed range: {:.1f} - {:.1f} kph".format(min(measured_speeds), max(measured_speeds)))
    
    # Simulate control loop
    conditions_seen = set()
    
    for i in range(len(t)):
        ref_speed = ref_speeds[i] 
        measured_speed = measured_speeds[i]
        
        # Simulate tracking error and control effort
        tracking_error = ref_speed - measured_speed
        control_effort = np.random.uniform(-10, 80)  # PWM %
        solve_time = np.random.uniform(5, 25)        # ms
        success = np.random.random() > 0.05          # 95% success rate
        cost = abs(tracking_error) * 10 + np.random.uniform(0, 50)
        
        # Record performance
        tuner.record_performance(
            ref_speed, measured_speed, tracking_error, 
            control_effort, solve_time, success, cost
        )
        
        # Track conditions for statistics
        condition = tuner.classify_operating_condition(ref_speed, measured_speed)
        conditions_seen.add(condition)
        
        # Check for parameter updates
        if tuner.should_update_parameters():
            new_params = tuner.get_current_parameters()
            print(f"[{t[i]:6.1f}s] Parameter update in condition: {condition}")
            print(f"         New Q={new_params.Q_val:.1f}, R={new_params.R_val:.3f}")
        
        # Print progress occasionally
        if i % 100 == 0:
            print(f"[{t[i]:6.1f}s] Speed: {measured_speed:5.1f} kph, "
                  f"Ref: {ref_speed:5.1f} kph, Condition: {condition}")
    
    # Generate final report
    print("\n" + "=" * 60)
    print("ADAPTIVE TUNING TEST RESULTS")
    print("=" * 60)
    
    report = tuner.generate_adaptive_report()
    print(report)
    
    print(f"\nConditions encountered during test: {len(conditions_seen)}")
    for i, condition in enumerate(sorted(conditions_seen), 1):
        print(f"  {i}. {condition}")
    
    # Save results
    tuner.save_adaptive_results("test_adaptive_results.json")
    print("\nTest results saved to 'test_adaptive_results.json'")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_adaptive_tuning()