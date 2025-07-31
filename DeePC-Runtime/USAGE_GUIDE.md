# Fixed Hankel Matrix DeePC System - Usage Guide

## Overview

This system implements a **Fixed Hankel Matrix approach** for DeePC (Data-enabled Predictive Control) to achieve optimal WLTP cycle tracking. Instead of using a sliding window approach, it uses pre-computed Hankel matrices collected from PRBS (Pseudo-Random Binary Sequence) data at different speed operating points.

## Key Benefits

- **Optimal Performance**: Matrices are pre-optimized for each speed range
- **Computational Efficiency**: No online Hankel matrix computation
- **Better Tracking**: Speed-scheduled matrices provide better control performance
- **Robustness**: Quality-validated matrices ensure reliable operation

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ PRBS Data       │───▶│ Hankel Matrix   │───▶│ Speed-Scheduled │
│ Collection      │    │ Analysis &      │    │ Selection       │
│                 │    │ Optimization    │    │ System          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Validation &    │◀───│ Fixed Hankel    │◀───│ Fixed Hankel    │
│ Testing         │    │ DeePC           │    │ Acados Solver   │
│ Framework       │    │ Controller      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Step-by-Step Usage

### Phase 1: Data Collection

**Purpose**: Collect PRBS excitation data at different speed operating points.

```bash
cd DeePC-Runtime
python PRBSDataCollector.py
```

**What it does**:
- Collects PRBS data at 20 speed operating points (5-100 kph)
- Uses PID controller to maintain target speeds
- Applies PRBS excitation for system identification
- Saves data in `dataForHankle/PRBSCollection/`

**Duration**: ~2-3 hours (depending on SOC and collection parameters)

**Requirements**:
- Vehicle with sufficient SOC (>80% recommended)
- CAN bus communication operational
- PWM hardware connected and functional

### Phase 2: Hankel Matrix Generation

**Purpose**: Analyze collected data and generate optimal Hankel matrices.

```bash
python HankelMatrixAnalyzer.py
```

**What it does**:
- Loads PRBS collected data
- Analyzes data quality and richness
- Optimizes Hankel matrix dimensions for each operating point
- Generates quality-validated matrices
- Saves results in `dataForHankle/OptimizedMatrices/`

**Output**:
- Individual matrix files: `hankel_*kph_HHMM_MMDD.npz`
- Complete collection: `complete_hankel_collection_HHMM_MMDD.pkl`
- Analysis report and plots

### Phase 3: System Validation

**Purpose**: Validate the performance of the Fixed Hankel system.

```bash
python ValidationFramework.py
```

**What it does**:
- Tests Hankel matrix quality across speed range
- Validates solver performance and real-time capability
- Simulates WLTP tracking performance
- Compares with baseline methods
- Generates comprehensive validation report

### Phase 4: Production Usage

**Purpose**: Use the Fixed Hankel DeePC controller for optimal WLTP tracking.

```bash
python PIDDeePCControllerFixed.py
```

**What it does**:
- Loads pre-computed Hankel matrices
- Implements speed-scheduled matrix selection
- Runs 10Hz control loop with DeePC + PID fallback
- Automatically adapts matrices based on vehicle speed
- Logs performance data for analysis

## File Structure

```
DeePC-Runtime/
├── PRBSDataCollector.py          # Phase 1: PRBS data collection
├── HankelMatrixAnalyzer.py       # Phase 2: Matrix generation
├── SpeedScheduledHankel.py       # Matrix selection system
├── DeePCAcadosFixed.py          # Fixed Hankel Acados solver
├── PIDDeePCControllerFixed.py   # Phase 4: Production controller
├── ValidationFramework.py        # Phase 3: System validation
├── deepc_config.py              # Configuration parameters
├── utils_deepc.py               # Utility functions
└── USAGE_GUIDE.md               # This file

dataForHankle/
├── PRBSCollection/              # Collected PRBS data
├── OptimizedMatrices/           # Generated Hankel matrices
└── smallDataSet/                # Legacy data (optional)

Log_DriveRobot/                  # Controller performance logs
ValidationResults/               # Validation reports
```

## Configuration

### Key Parameters (deepc_config.py)

```python
# Control frequency
CONTROL_FREQUENCY = 10  # Hz

# DeePC parameters
MANUAL_DEEPC_PARAMS = DeePCParameters(
    Tini=25,                    # Initialization horizon
    THorizon=15,                # Prediction horizon
    hankel_subB_size=2000,      # Hankel matrix size
    Q_val=400.0,                # Output tracking weight
    R_val=0.08,                 # Control effort weight
    lambda_g_val=80.0,          # g regularization
    lambda_y_val=15.0,          # Output regularization
    lambda_u_val=15.0,          # Input regularization
    decay_rate_q=0.85,          # Q decay rate
    decay_rate_r=0.12           # R decay rate
)

# System constraints
U_MIN = -100.0                  # Minimum control (%)
U_MAX = 100.0                   # Maximum control (%)
Y_MIN = 0.0                     # Minimum speed (kph)
Y_MAX = 150.0                   # Maximum speed (kph)
```

### Hardware Configuration

```python
# CAN interfaces
DYNO_CAN_INTERFACE = 'can0'     # Speed/force data
VEH_CAN_INTERFACE = 'can1'      # Vehicle SOC data

# PWM hardware
CP2112_BUS = 3                  # I2C bus for PWM board
```

## Advanced Usage

### Custom Operating Points

Modify speed operating points in `PRBSDataCollector.py`:

```python
self.speed_operating_points = [
    10, 20, 30, 40, 50,
    60, 70, 80, 90, 100,
    110, 120  # Extended range
]
```

### Matrix Selection Tuning

Adjust matrix selection behavior in `SpeedScheduledHankel.py`:

```python
self.config = {
    'interpolation_method': 'linear',    # 'linear', 'nearest', 'rbf'
    'update_threshold': 2.0,             # Speed change threshold (kph)
    'use_matrix_blending': True,         # Enable smooth transitions
    'blend_window': 5.0,                 # Blending window (kph)
}
```

### Solver Optimization

Tune Acados solver settings in `DeePCAcadosFixed.py`:

```python
ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
ocp.solver_options.nlp_solver_type = 'SQP_RTI'
ocp.solver_options.print_level = 0           # Minimize output
ocp.solver_options.qp_solver_warm_start = 1  # Enable warm start
```

## Troubleshooting

### Common Issues

1. **"No Hankel matrix files found"**
   - Run phases in correct order: Collection → Analysis → Validation → Usage
   - Check file paths and permissions

2. **"Solver compilation failed"**
   - Ensure Acados is properly installed
   - Check C compiler availability
   - Clear `c_generated_code` directory and retry

3. **"CAN communication failed"**
   - Verify CAN interfaces are up and configured
   - Check DBC file paths
   - Ensure proper CAN bitrate (500000)

4. **"Poor tracking performance"**
   - Check data quality in validation report
   - Consider recollecting PRBS data
   - Tune DeePC parameters

5. **"Real-time violations"**
   - Reduce problem size (smaller Tini/THorizon)
   - Optimize solver settings
   - Check system load

### Performance Optimization

1. **Improve Data Quality**:
   - Collect more PRBS data
   - Use higher excitation amplitude
   - Ensure good speed tracking during collection

2. **Reduce Computational Load**:
   - Use smaller matrix dimensions
   - Reduce number of operating points
   - Optimize solver settings

3. **Better Tracking**:
   - Tune cost function weights (Q, R, λ)
   - Adjust time-varying decay rates
   - Fine-tune matrix selection thresholds

## Monitoring and Maintenance

### Performance Metrics

Monitor these key metrics during operation:

- **DeePC Success Rate**: Should be >90%
- **Average Solve Time**: Should be <50ms for 10Hz operation
- **Tracking RMSE**: Should be <2 kph for WLTP
- **Matrix Update Frequency**: Typically 5-10 per cycle

### Log Analysis

Performance logs contain detailed information:

```python
# Load and analyze logs
df = pd.read_excel('Log_DriveRobot/latest_log.xlsx')

# Key metrics
deepc_success_rate = df['deepc_success'].mean()
avg_solve_time = df['solve_time_ms'].mean()
tracking_rmse = np.sqrt(np.mean(df['error']**2))
```

### Periodic Maintenance

- **Weekly**: Check performance logs and success rates
- **Monthly**: Validate against new drive cycles
- **Quarterly**: Consider recollecting data if performance degrades
- **Annually**: Full system validation and parameter tuning

## Support and Development

### Getting Help

1. Check validation reports for system health
2. Review log files for error patterns
3. Run diagnostic tests using validation framework
4. Consult configuration guide for parameter tuning

### Contributing Improvements

1. Add new operating points for extended speed ranges
2. Implement additional matrix interpolation methods
3. Optimize solver configurations for specific hardware
4. Enhance validation metrics and tests

### Future Enhancements

- Adaptive online matrix learning
- Multi-objective optimization for comfort/efficiency
- Integration with vehicle state estimation
- Advanced fault detection and diagnosis

---

## Quick Start Checklist

- [ ] Phase 1: Collect PRBS data (`python PRBSDataCollector.py`)
- [ ] Phase 2: Generate matrices (`python HankelMatrixAnalyzer.py`)
- [ ] Phase 3: Run validation (`python ValidationFramework.py`)
- [ ] Phase 4: Deploy controller (`python Enhanced-Controller.py`)
- [ ] Monitor performance and tune as needed

For optimal WLTP tracking, ensure all phases complete successfully before production deployment!