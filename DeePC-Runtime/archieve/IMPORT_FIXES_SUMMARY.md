# Import Fixes After Folder Reorganization

## Summary
Fixed all import errors that occurred after reorganizing the folder structure into core/, utils/, and data_collection/ directories.

## Files Fixed

### 1. core/PIDDeePCControllerFixed.py
- Changed `from utils.utils import *` to `from utils import *`
- Changed `from core.DeePCAcadosFixed` to `from DeePCAcadosFixed`
- Changed `from utils.deepc_config` to `from deepc_config`

### 2. core/DeePCAcadosFixed.py
- Changed `from .SpeedScheduledHankel` to `from SpeedScheduledHankel`

### 3. data_collection/PRBSDataCollector.py
- Changed `from utils.utils import *` to `from utils import *`
- Changed `from utils.deepc_config` to `from deepc_config`

### 4. core/EnhancedPID-Controller-IWR.py
- Changed `from utils.utils import *` to `from utils import *`
- Added explicit `from smbus2 import SMBus` for hardware access

### 5. run_enhanced_pid.py
- Completely rewrote to use subprocess.run() to handle the hyphenated filename
- This approach avoids Python module naming restrictions

## Testing Results

All entry points now work correctly when run with the conda environment Python:
```bash
/home/guiliang/anaconda3/envs/drivingRobot/bin/python run_controller.py
/home/guiliang/anaconda3/envs/drivingRobot/bin/python run_data_collection.py
/home/guiliang/anaconda3/envs/drivingRobot/bin/python run_enhanced_pid.py
```

The applications start successfully but may fail due to:
- Missing Hankel matrix files (need to run data collection first)
- Missing hardware (I2C bus, CAN interfaces)

These are expected failures in a non-hardware environment.

## Key Lessons
1. When reorganizing Python projects, relative imports within packages need adjustment
2. Python module names cannot contain hyphens - use subprocess for such files
3. Always test all entry points after reorganization
4. The sys.path modifications in entry scripts help ensure imports work correctly