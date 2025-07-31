# DeePC-Runtime: Advanced Data-Enabled Predictive Control System

## Overview

DeePC-Runtime is a state-of-the-art control system that combines Data-Enabled Predictive Control (DeePC) with an enhanced PID controller for optimal vehicle speed tracking. The system features online parameter autotuning, comprehensive performance monitoring, and real-time metrics tracking to achieve superior control performance.

## Key Features

- **Fixed-Horizon DeePC**: Optimized with Tini=20 and THorizon=20 for consistent real-time performance
- **Online Parameter Autotuning**: Continuously optimizes Q, R, λg, λy, λu parameters based on operating conditions
- **Enhanced PID Fallback**: Additive enhancement architecture with adaptive gains and real-time learning
- **Comprehensive Performance Monitoring**: Real-time tracking of RMSSE, IWR, DTI, and error rates
- **Speed-Scheduled Hankel Matrices**: Pre-computed matrices for different operating points
- **Visual Performance Indicators**: Clear feedback with ✓/✗ symbols for target achievement
- **Condition-Based Learning**: Adapts to different speed ranges and acceleration/deceleration patterns

## Project Structure

```
DeePC-Runtime/
├── core/                              # Core controllers and solvers
│   ├── PIDDeePCControllerFixed.py     # Main DeePC controller with autotuning
│   ├── EnhancedPID-Controller-IWR.py  # Enhanced PID with IWR tracking
│   ├── DeePCAcadosFixed.py            # Acados-based DeePC solver
│   ├── DeePCCVXPYSolver.py            # CVXPY-based solver (backup)
│   └── SpeedScheduledHankel.py        # Speed-scheduled matrix manager
│
├── utils/                             # Utilities and configuration
│   ├── utils.py                       # Core utility functions
│   ├── deepc_config.py                # System configuration
│   └── DeePCParametersSimple.py       # Parameter management
│
├── data_collection/                   # Data collection tools
│   ├── PRBSDataCollector.py           # PRBS data collection
│   └── HankelMatrixAnalyzer.py        # Hankel matrix analysis
│
├── archieve/                          # Previous versions (reference only)
│
├── dataForHankle/                     # Hankel matrix storage
├── Log_DriveRobot/                    # Control logs and results
│
├── run_controller.py                  # Main entry point
├── run_data_collection.py             # Data collection entry point
├── run_enhanced_pid.py                # Standalone PID entry point
└── README.md                          # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Conda environment (recommended)
- CAN interface for vehicle communication
- CP2112 USB-to-I2C bridge

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd DeePC-Runtime
   ```

2. Create conda environment:
   ```bash
   conda create -n drivingRobot python=3.10
   conda activate drivingRobot
   ```

3. Install dependencies:
   ```bash
   pip install numpy==1.26.4  # Important: Use NumPy 1.x for compatibility
   pip install scipy pandas matplotlib
   pip install casadi cvxpy cantools python-can smbus2
   pip install acados  # Optional: for Acados solver
   ```

4. Configure CAN interfaces:
   ```bash
   sudo ip link set can0 type can bitrate 500000
   sudo ip link set can0 up
   sudo ip link set can1 type can bitrate 500000
   sudo ip link set can1 up
   ```

## Usage

### 1. Data Collection (First Time Setup)

Collect PRBS data at different operating points to build Hankel matrices:

```bash
python run_data_collection.py
```

This will:
- Apply PRBS excitation at speeds: 5, 10, 15, ..., 100 kph
- Collect 60 seconds of data per operating point
- Save data to `dataForHankle/PRBS_Collection_YYYYMMDD_HHMMSS/`

### 2. Hankel Matrix Analysis

Analyze collected data and create optimized Hankel matrices:

```bash
cd data_collection
python HankelMatrixAnalyzer.py
```

This will:
- Process PRBS data from the latest collection
- Compute optimal Hankel matrices for each speed range
- Perform quality assessment and validation
- Save matrices to `dataForHankle/OptimizedMatrices/`

### 3. Run Main DeePC Controller

Execute the main control loop with online autotuning:

```bash
python Enhanced-Controller.py
```

## Control Architecture

### 1. DeePC with Fixed Hankel Matrices
- **Fixed Parameters**: Tini=20, THorizon=20 for consistent performance
- **Speed-Scheduled Matrices**: Pre-computed matrices selected based on current speed
- **Intelligent Interpolation**: Smooth transitions between speed ranges
- **Fallback Logic**: Automatic switch to enhanced PID if DeePC fails

### 2. PID Controller
**Enhancement Components:**
- **Adaptive Gain Enhancement**: ±8% gain adjustment based on performance
- **Data-Driven Feedforward**: Pattern-based feedforward corrections
- **Error Pattern Learning**: Speed-range specific error corrections
- **Performance-Based Scaling**: Dynamic enhancement strength (0.3-1.2x)

### 3. Performance Monitoring

**Real-time Metrics:**
- **RMSSE**: Root Mean Square Speed Error (target < 1.2 kph)
- **IWR**: Idle Waste Rate (target < 1.0%)
- **Error Rate**: Percentage with |error| > 1.5 kph (target < 5%)
- **DTI**: Drive Trace Index (0-100 composite score)

**DTI Calculation:**
```
DTI = 0.4 × IWR_score + 0.4 × RMSSE_score + 0.2 × ErrorRate_score

Where:
- IWR_score = max(0, 100 - IWR × 10)
- RMSSE_score = max(0, 100 - RMSSE × 50)  
- ErrorRate_score = max(0, 100 - ErrorRate × 5)
```

## Configuration

Edit `utils/deepc_config.py` to adjust system parameters:

```python
# Control Parameters
CONTROL_FREQUENCY = 10  # Hz
U_MIN, U_MAX = -30, 100  # PWM limits (%)
Y_MIN, Y_MAX = 0, 130    # Speed limits (kph)

# Initial DeePC Parameters (will be optimized online)
MANUAL_DEEPC_PARAMS = DeePCParameters(
    Tini=20,                  # Fixed for autotuning
    THorizon=20,              # Fixed for autotuning
    Q_val=400.0,              # Initial output weight
    R_val=0.08,               # Initial control weight
    lambda_g_val=80.0,        # Initial g regularization
    lambda_y_val=15.0,        # Initial y tolerance
    lambda_u_val=15.0,        # Initial u tolerance
    decay_rate_q=0.85,        # Q decay rate
    decay_rate_r=0.12         # R decay rate
)
```

## Real-Time Output

### Console Display
```
[100.5s] ✓ v_ref= 45.2, v_meas= 44.8, e=+0.4 kph | u= 35.2%, FixedDeePC | RMSSE=0.52, IWR=0.8%, MaxErr=1.2 | Tuner:MONITORING Q=450,R=0.075
```

### Performance Summary
```
================================================================================
[Controller] Cycle 'WLTP' Completed - Performance Summary
================================================================================

[Control Statistics]
  Total cycles: 1800
  DeePC activations: 1710 (95.0%)
  PID fallback activations: 90
  Matrix updates: 45
  Average solve time: 12.34ms
  Success rate: 95.0%

[Performance Metrics]
  RMSSE: 0.821 kph ✓ (target < 1.2)
  IWR: 0.92% ✓ (target < 1.0%)
  Error Rate >1.5kph: 3.2% ✓ (target < 5%)
  DTI Score: 87.5/100

[Target Achievement: 3/3 met]

[Autotuner Summary]
  Best parameters by condition:
    low_speed: Q=380, R=0.095, λg=75, λy=18, λu=20 (score=0.412)
    medium_speed: Q=425, R=0.082, λg=85, λy=16, λu=17 (score=0.385)
    high_speed: Q=480, R=0.070, λg=90, λy=14, λu=15 (score=0.398)
    acceleration: Q=520, R=0.065, λg=95, λy=12, λu=13 (score=0.425)
    deceleration: Q=350, R=0.110, λg=70, λy=22, λu=25 (score=0.445)
================================================================================
```

## Data Logging

All data is saved to Excel files in `Log_DriveRobot/` with naming format:
```
HHMM_MMDD_DR_log_<vehicle>_<cycle>_Start<SOC>%_<algorithm>_Ts<period>_Q<value>_R<value>_FixedHankel.xlsx
```

Logged data includes:
- Time-series: time, reference/measured speed, control output, errors
- Controller info: type (DeePC/PID), solve time, cost
- Real-time metrics: RMSSE, IWR, max error at each timestep
- Final metrics: DTI components and scores
- Autotuner parameters: best values for each condition

## Troubleshooting

### Common Issues

1. **NumPy Compatibility Error**
   ```bash
   pip install "numpy<2" --force-reinstall
   ```

2. **CAN Interface Not Found**
   ```bash
   # Check CAN setup
   ip link show can0
   sudo ip link set can0 up
   ```

3. **No Hankel Matrices Found**
   - Run `python run_data_collection.py` first
   - Then run `HankelMatrixAnalyzer.py`

4. **Import Errors**
   - Ensure you run scripts from the DeePC-Runtime root directory
   - Or use the provided entry point scripts (run_*.py)

5. **DeePC Solver Failures**
   - Check constraint violations in logs
   - System automatically falls back to enhanced PID
   - Autotuner will adapt parameters to improve success rate

## Advanced Usage

### Disable Autotuning
In `core/PIDDeePCControllerFixed.py`:
```python
self.enable_autotuning = False  # Line ~929
```

### Adjust Autotuning Parameters
In `core/PIDDeePCControllerFixed.py`, modify the OnlineAutotuner class:
```python
self.update_interval = 200      # Update every 20 seconds
self.exploration_rate = 0.10    # 10% exploration
self.adaptation_rate = 0.03     # 3% change rate
```

### Change Performance Weights
In the `calculate_performance_score` method:
```python
rmsse_score = rmsse * 0.7       # 70% weight on tracking
effort_score = control_effort * 0.001 * 0.2  # 20% on effort
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit with clear messages: `git commit -am 'Add feature'`
5. Push branch: `git push origin feature-name`
6. Submit pull request with description

## License


## Authors

- Enhanced PID Controller: Guiliang Zheng
- DeePC Implementation: Guiliang Zheng
- Online Autotuning: Guiliang Zheng

## References

1. Coulson, J., et al. "Data-enabled predictive control: In the shallows of the DeePC." 2019 European Control Conference (ECC). IEEE, 2019.
2. Speed-scheduled control strategies for autonomous vehicles
3. Adaptive PID control with enhancement layers
4. Online parameter tuning for model predictive control