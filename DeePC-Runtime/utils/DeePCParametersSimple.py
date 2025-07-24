#!/usr/bin/env python3
"""
Simple DeePCParameters class without type annotations
Compatible with older Python versions
"""

class DeePCParameters:
    """Class to hold all DeePC tunable parameters"""
    def __init__(self, **kwargs):
        # Core DeePC parameters
        self.Tini = kwargs.get('Tini', 30)
        self.THorizon = kwargs.get('THorizon', 30)
        self.hankel_subB_size = kwargs.get('hankel_subB_size', 120)
        
        # Weighting parameters
        self.Q_val = kwargs.get('Q_val', 265.0)
        self.R_val = kwargs.get('R_val', 0.15)
        self.lambda_g_val = kwargs.get('lambda_g_val', 60.0)
        self.lambda_y_val = kwargs.get('lambda_y_val', 10.0)
        self.lambda_u_val = kwargs.get('lambda_u_val', 10.0)
        
        # Decay parameters for time-varying weights
        self.decay_rate_q = kwargs.get('decay_rate_q', 0.9)
        self.decay_rate_r = kwargs.get('decay_rate_r', 0.1)
        
        # Solver parameters
        self.solver_type = kwargs.get('solver_type', "acados")
        self.max_iterations = kwargs.get('max_iterations', 50)
        self.tolerance = kwargs.get('tolerance', 1e-5)
    
    def copy(self):
        """Create a copy of the parameters"""
        return DeePCParameters(
            Tini=self.Tini,
            THorizon=self.THorizon,
            hankel_subB_size=self.hankel_subB_size,
            Q_val=self.Q_val,
            R_val=self.R_val,
            lambda_g_val=self.lambda_g_val,
            lambda_y_val=self.lambda_y_val,
            lambda_u_val=self.lambda_u_val,
            decay_rate_q=self.decay_rate_q,
            decay_rate_r=self.decay_rate_r,
            solver_type=self.solver_type,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance
        )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'Tini': self.Tini,
            'THorizon': self.THorizon,
            'hankel_subB_size': self.hankel_subB_size,
            'Q_val': self.Q_val,
            'R_val': self.R_val,
            'lambda_g_val': self.lambda_g_val,
            'lambda_y_val': self.lambda_y_val,
            'lambda_u_val': self.lambda_u_val,
            'decay_rate_q': self.decay_rate_q,
            'decay_rate_r': self.decay_rate_r,
            'solver_type': self.solver_type,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(**data)


class PerformanceMetrics:
    """Class to hold performance evaluation metrics"""
    def __init__(self):
        self.tracking_error_rms = float('inf')
        self.tracking_error_max = float('inf')
        self.control_effort_rms = float('inf')
        self.control_smoothness = float('inf')
        self.solver_time_avg = float('inf')
        self.solver_success_rate = 0.0
        self.overall_score = float('inf')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'tracking_error_rms': self.tracking_error_rms,
            'tracking_error_max': self.tracking_error_max,
            'control_effort_rms': self.control_effort_rms,
            'control_smoothness': self.control_smoothness,
            'solver_time_avg': self.solver_time_avg,
            'solver_success_rate': self.solver_success_rate,
            'overall_score': self.overall_score
        }


# Factory function for backward compatibility
def create_deepc_parameters(**kwargs):
    """Create DeePCParameters with default values"""
    return DeePCParameters(**kwargs)