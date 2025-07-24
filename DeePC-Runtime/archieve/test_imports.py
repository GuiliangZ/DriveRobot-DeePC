#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    from deepc_config import *
    print("OK deepc_config imported successfully")
    print("  PID_KP: {}".format(PID_KP))
    print("  Control frequency: {}".format(CONTROL_FREQUENCY))
except Exception as e:
    print("FAIL deepc_config import failed: {}".format(e))

try:
    from utils_deepc import compute_pid_control, get_gains_for_speed
    print("OK utils_deepc functions imported successfully")
    
    # Test get_gains_for_speed
    kp, ki, kd, kff = get_gains_for_speed(50.0)
    print("  PID gains at 50 kph: Kp={}, Ki={}, Kd={}, Kff={}".format(kp, ki, kd, kff))
except Exception as e:
    print("FAIL utils_deepc import failed: {}".format(e))

try:
    from DeePCParametersSimple import DeePCParameters
    params = DeePCParameters(Tini=20, THorizon=15)
    print("OK DeePCParametersSimple imported successfully")
    print("  Test params: Tini={}, THorizon={}".format(params.Tini, params.THorizon))
except Exception as e:
    print("FAIL DeePCParametersSimple import failed: {}".format(e))

try:
    from PRBSDataCollector import PRBSDataCollector
    print("OK PRBSDataCollector imported successfully")
except Exception as e:
    print("FAIL PRBSDataCollector import failed: {}".format(e))

try:
    from PIDDeePCControllerFixed import FixedHankelDeePCController
    print("OK PIDDeePCControllerFixed imported successfully")
except Exception as e:
    print("FAIL PIDDeePCControllerFixed import failed: {}".format(e))

print("\nImport test completed!")