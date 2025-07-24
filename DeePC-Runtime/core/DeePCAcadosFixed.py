"""
Name: DeePCAcadosFixed.py
Author: Modified for Fixed Hankel Matrix System
Date: 2025-07-21
Version: 2.0.0
Description: Enhanced Acados-based DeePC solver using fixed pre-computed Hankel matrices
            for optimal performance across different speed operating points.
"""
import time
import warnings
import numpy as np
from scipy.linalg import block_diag
import casadi as cs
from casadi import vcat
import casadi.tools as ctools
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# Import speed-scheduled Hankel system
from SpeedScheduledHankel import SpeedScheduledHankel

def timer(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        print('Time elapsed: {}'.format(time.time() - start))
        return ret
    return wrapper


class DeePCFixedHankelSolver:
    """
    Fixed Hankel Matrix DeePC solver with speed scheduling for optimal WLTP tracking.
    Uses pre-computed Hankel matrices instead of sliding window approach.
    """

    def __init__(self, u_dim=1, y_dim=1, hankel_data_file=None, ineqconidx=None, ineqconbd=None):
        """
        Initialize DeePC solver with fixed Hankel matrices.
        
        Args:
            u_dim: Input dimension (1 for SISO)
            y_dim: Output dimension (1 for SISO)
            hankel_data_file: Path to pre-computed Hankel matrices
            ineqconidx: Inequality constraint indices
            ineqconbd: Inequality constraint bounds
        """
        self.u_dim = u_dim
        self.y_dim = y_dim
        
        # Initialize speed-scheduled Hankel system
        self.hankel_scheduler = SpeedScheduledHankel(hankel_data_file)
        
        # Get system information
        system_info = self.hankel_scheduler.get_system_info()
        if system_info['status'] != 'Ready':
            raise RuntimeError(f"Hankel system not ready: {system_info['status']}")
        
        # Extract consistent parameters across all operating points
        self.param_ranges = system_info['parameter_ranges']
        self.operating_points = system_info['operating_points']
        
        # Use maximum dimensions to ensure compatibility
        self.Tini_max = self.param_ranges['Tini'][1]
        self.THorizon_max = self.param_ranges['THorizon'][1]
        self.g_dim_max = self.param_ranges['g_dim'][1]
        
        print(f"[FixedDeePC] Initialized with {len(self.operating_points)} operating points")
        print(f"[FixedDeePC] Max dimensions: Tini={self.Tini_max}, THorizon={self.THorizon_max}, g_dim={self.g_dim_max}")
        
        # Store constraint information
        self.ineqconidx = ineqconidx
        self.ineqconbd = ineqconbd
        
        # Current state
        self.current_matrices = None
        self.current_params = None
        self.solver = None
        self.last_speed = None
        
        # Initialize with default matrices (use first operating point)
        default_speed = self.operating_points[len(self.operating_points)//2]  # Use middle speed as default
        self.current_matrices = self.hankel_scheduler.get_matrices_for_speed(default_speed)
        self.current_params = self.current_matrices['params']
        
        # Setup CasADi variables for the maximum dimensions
        self._init_variables()
        
    def _init_variables(self):
        """Initialize CasADi variables for maximum expected dimensions."""
        # Use maximum g_dim for consistency across all operating points
        self.g_dim = self.g_dim_max
        
        # Decision variable (fixed size based on maximum g_dim)
        self.optimizing_target = ctools.struct_symSX([
            (ctools.entry('g', shape=(self.g_dim, 1)))
        ])
        
        # Parameters (use maximum dimensions for consistency)
        parameters = [
            ctools.entry('uini', shape=(self.u_dim * self.Tini_max, 1)),
            ctools.entry('yini', shape=(self.y_dim * self.Tini_max, 1)),
            ctools.entry('yref', shape=(self.y_dim * self.THorizon_max, 1)),
            ctools.entry('Up',   shape=(self.u_dim * self.Tini_max, self.g_dim)),
            ctools.entry('Yp',   shape=(self.y_dim * self.Tini_max, self.g_dim)),
            ctools.entry('Uf',   shape=(self.u_dim * self.THorizon_max, self.g_dim)),
            ctools.entry('Yf',   shape=(self.y_dim * self.THorizon_max, self.g_dim)),
            ctools.entry('lambda_g', shape=(self.g_dim, self.g_dim)),
            ctools.entry('lambda_y', shape=(self.y_dim * self.Tini_max, self.y_dim * self.Tini_max)),
            ctools.entry('lambda_u', shape=(self.u_dim * self.Tini_max, self.u_dim * self.Tini_max)),
            ctools.entry('Q', shape=(self.y_dim * self.THorizon_max, self.y_dim * self.THorizon_max)),
            ctools.entry('R', shape=(self.u_dim * self.THorizon_max, self.u_dim * self.THorizon_max)),
            # Active dimensions (to handle variable-size matrices)
            ctools.entry('active_Tini', shape=(1, 1)),
            ctools.entry('active_THorizon', shape=(1, 1)),
        ]
        
        self.parameters = ctools.struct_symSX(parameters)

    def _init_ineq_cons(self, ineqconidx, ineqconbd, Uf, Yf, du, active_THorizon):
        """Initialize inequality constraints with active horizon consideration."""
        if ineqconidx is None:
            return [], [], [], False
        
        Hc_list = []
        lbc_list = []
        ubc_list = []
        
        for varname, idx in ineqconidx.items():
            if varname == 'u':
                H_all = Uf
                dim = self.u_dim
                lb = ineqconbd['lbu']
                ub = ineqconbd['ubu']
            elif varname == 'y':
                H_all = Yf
                dim = self.y_dim
                lb = ineqconbd['lby']
                ub = ineqconbd['uby']
            elif varname == 'du':
                continue  # Handle separately
            else:
                raise ValueError(f"Variable {varname} not supported")
            
            # Only use active horizon
            active_THorizon_int = int(active_THorizon)
            idx_H = [v + i * dim for i in range(active_THorizon_int) for v in idx]
            
            Hc_list.append(H_all[idx_H, :])
            lbc_list.append(np.tile(lb, active_THorizon_int))
            ubc_list.append(np.tile(ub, active_THorizon_int))
        
        if Hc_list:
            Hc = cs.vertcat(*Hc_list)
            lbc = np.concatenate(lbc_list).flatten().tolist()
            ubc = np.concatenate(ubc_list).flatten().tolist()
            ineq_flag = True
        else:
            Hc, lbc, ubc = [], [], []
            ineq_flag = False
        
        return Hc, lbc, ubc, ineq_flag

    @timer
    def init_acados_solver(self, recompile_solver=True):
        """
        Initialize Acados solver with fixed maximum dimensions.
        """
        print('[FixedDeePC] Initializing Acados solver with fixed Hankel approach...')
        
        if self.g_dim <= (self.u_dim + self.y_dim) * self.Tini_max:
            raise ValueError(f'Insufficient degrees of freedom: {self.g_dim} <= {(self.u_dim + self.y_dim) * self.Tini_max}')
        
        # Get symbolic parameters
        params = self.parameters[...]
        uini, yini, yref = params['uini'], params['yini'], params['yref']
        Up_cur, Yp_cur, Uf_cur, Yf_cur = params['Up'], params['Yp'], params['Uf'], params['Yf']
        lambda_g, lambda_y, lambda_u = params['lambda_g'], params['lambda_y'], params['lambda_u']
        Q, R = params['Q'], params['R']
        active_Tini, active_THorizon = params['active_Tini'], params['active_THorizon']
        
        g, = self.optimizing_target[...]
        
        # Control calculations
        u_cur = cs.mtimes(Uf_cur, g)
        u_prev = cs.vertcat(uini[-self.u_dim:], u_cur[:-self.u_dim])
        du = u_cur - u_prev
        
        # Setup constraints with active dimensions
        Hc, lbc_ineq, ubc_ineq, ineq_flag = self._init_ineq_cons(
            self.ineqconidx, self.ineqconbd, Uf_cur, Yf_cur, du, self.THorizon_max
        )
        
        # Handle du constraints
        du_flag = False
        if self.ineqconidx is not None and 'du' in self.ineqconidx:
            idx = self.ineqconidx['du']
            if idx:
                idx_H = [v + i * self.u_dim for i in range(self.THorizon_max) for v in idx]
                h_du = du[idx_H]
                lbdu = self.ineqconbd['lbdu']
                ubdu = self.ineqconbd['ubdu']
                lbc_du = np.tile(lbdu, self.THorizon_max).flatten().tolist()
                ubc_du = np.tile(ubdu, self.THorizon_max).flatten().tolist()
                du_flag = True
        
        # Acados model setup
        model = AcadosModel()
        model.name = 'deepc_fixed'
        model.x = g
        model.p = cs.vertcat(
            uini, yini, yref,
            cs.reshape(Up_cur, -1, 1), cs.reshape(Yp_cur, -1, 1),
            cs.reshape(Uf_cur, -1, 1), cs.reshape(Yf_cur, -1, 1),
            cs.reshape(Q, -1, 1), cs.reshape(R, -1, 1),
            cs.reshape(lambda_y, -1, 1), cs.reshape(lambda_g, -1, 1), 
            cs.reshape(lambda_u, -1, 1),
            active_Tini, active_THorizon
        )
        model.disc_dyn_expr = model.x  # Static optimization
        
        # Objective function with active dimension masking
        r1 = Yf_cur @ g - yref  # Tracking error
        r2 = Uf_cur @ g         # Control effort
        r3 = Yp_cur @ g - yini  # Init condition slack Y
        r4 = Up_cur @ g - uini  # Init condition slack U
        r5 = g                  # Regularization
        
        # Objective with active dimension weighting
        obj = r1.T @ Q @ r1 + r2.T @ R @ r2 + r3.T @ lambda_y @ r3 + r4.T @ lambda_u @ r4 + r5.T @ lambda_g @ r5
        
        # OCP setup
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = 1
        ocp.dims.nx = self.g_dim
        ocp.dims.np = model.p.size()[0]
        
        # Cost setup
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        model.cost_expr_ext_cost = obj
        model.cost_expr_ext_cost_e = obj
        
        # Equality constraint: Up g = uini
        h_eq = cs.reshape(Up_cur @ g - uini, (-1, 1))
        
        # Setup inequality constraints
        has_constraints = ineq_flag or du_flag
        if has_constraints:
            ocp.constraints.constr_type = 'BGH'
            if ineq_flag:
                h1 = cs.reshape(Hc @ g, (-1, 1))
                lh_ineq = np.array(lbc_ineq)
                uh_ineq = np.array(ubc_ineq)
                if du_flag:
                    h = cs.vertcat(h1, cs.reshape(h_du, (-1, 1)))
                    lh_arr = np.concatenate((lh_ineq, np.array(lbc_du)))
                    uh_arr = np.concatenate((uh_ineq, np.array(ubc_du)))
                else:
                    h = h1
                    lh_arr = lh_ineq
                    uh_arr = uh_ineq
            else:
                h = cs.reshape(h_du, (-1, 1))
                lh_arr = np.array(lbc_du)
                uh_arr = np.array(ubc_du)
            
            ocp.dims.nh = h.shape[0]
            ocp.model.con_h_expr = h
            ocp.constraints.lh = lh_arr
            ocp.constraints.uh = uh_arr
        else:
            ocp.constraints.constr_type = 'BGH'
            ocp.model.con_h_expr = cs.SX.zeros(0, 1)
            ocp.constraints.lh = np.zeros(0)
            ocp.constraints.uh = np.zeros(0)
        
        # Solver options optimized for real-time performance
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = 'DISCRETE'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.levenberg_marquardt = 1e-4
        ocp.solver_options.tf = 1.0
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.print_level = 0
        
        # Initialize parameters
        n_p = ocp.model.p.numel()
        ocp.parameter_values = np.zeros(n_p)
        
        # Create solver
        if recompile_solver:
            self.solver = AcadosOcpSolver(
                ocp,
                json_file='DeePC_fixed_acados_ocp.json',
                generate=True,
                build=True,
                verbose=True,
            )
        else:
            self.solver = AcadosOcpSolver(
                None,
                json_file='DeePC_fixed_acados_ocp.json',
                generate=False,
                build=False,
                verbose=True,
            )
        
        print('[FixedDeePC] Acados solver initialized successfully')
    
    def update_matrices_for_speed(self, current_speed, force_update=False):
        """
        Update Hankel matrices based on current speed.
        
        Args:
            current_speed: Current vehicle speed (kph)
            force_update: Force matrix update regardless of threshold
            
        Returns:
            bool: True if matrices were updated
        """
        # Check if update is needed
        if not force_update and self.last_speed is not None:
            speed_change = abs(current_speed - self.last_speed)
            if speed_change < 2.0:  # 2 kph threshold
                return False
        
        # Get matrices for current speed
        matrices = self.hankel_scheduler.get_matrices_for_speed(current_speed)
        
        # Update current matrices
        self.current_matrices = matrices
        self.current_params = matrices['params']
        self.last_speed = current_speed
        
        return True
    
    def _prepare_matrices_for_solver(self, matrices, active_Tini, active_THorizon):
        """
        Prepare matrices for solver by padding to maximum dimensions if needed.
        
        Args:
            matrices: Current Hankel matrices
            active_Tini: Active initialization length
            active_THorizon: Active prediction horizon
            
        Returns:
            dict: Padded matrices ready for solver
        """
        Up, Uf, Yp, Yf = matrices['Up'], matrices['Uf'], matrices['Yp'], matrices['Yf']
        
        # Check if padding is needed
        if (Up.shape[0] == self.u_dim * self.Tini_max and 
            Yp.shape[0] == self.y_dim * self.Tini_max and
            Uf.shape[0] == self.u_dim * self.THorizon_max and
            Yf.shape[0] == self.y_dim * self.THorizon_max and
            Up.shape[1] == self.g_dim):
            # No padding needed
            return {
                'Up': Up, 'Uf': Uf, 'Yp': Yp, 'Yf': Yf
            }
        
        # Pad matrices to maximum dimensions
        Up_padded = np.zeros((self.u_dim * self.Tini_max, self.g_dim))
        Yp_padded = np.zeros((self.y_dim * self.Tini_max, self.g_dim))
        Uf_padded = np.zeros((self.u_dim * self.THorizon_max, self.g_dim))
        Yf_padded = np.zeros((self.y_dim * self.THorizon_max, self.g_dim))
        
        # Copy active portions
        Up_padded[:Up.shape[0], :min(Up.shape[1], self.g_dim)] = Up[:, :min(Up.shape[1], self.g_dim)]
        Yp_padded[:Yp.shape[0], :min(Yp.shape[1], self.g_dim)] = Yp[:, :min(Yp.shape[1], self.g_dim)]
        Uf_padded[:Uf.shape[0], :min(Uf.shape[1], self.g_dim)] = Uf[:, :min(Uf.shape[1], self.g_dim)]
        Yf_padded[:Yf.shape[0], :min(Yf.shape[1], self.g_dim)] = Yf[:, :min(Yf.shape[1], self.g_dim)]
        
        return {
            'Up': Up_padded, 'Uf': Uf_padded, 
            'Yp': Yp_padded, 'Yf': Yf_padded
        }
    
    def solve_step(self, uini, yini, yref, current_speed, Q_val, R_val, 
                   lambda_g_val, lambda_y_val, lambda_u_val, g_prev=None):
        """
        Solve one step of DeePC optimization with speed-scheduled matrices.
        
        Args:
            uini: Initial control history
            yini: Initial output history  
            yref: Reference trajectory
            current_speed: Current vehicle speed (kph)
            Q_val, R_val: Cost matrices
            lambda_g_val, lambda_y_val, lambda_u_val: Regularization matrices
            g_prev: Previous g solution for warm start
            
        Returns:
            tuple: (u_opt, g_opt, solve_time, feasible, cost)
        """
        if self.solver is None:
            raise RuntimeError("Solver not initialized. Call init_acados_solver() first.")
        
        # Update matrices for current speed
        self.update_matrices_for_speed(current_speed)
        
        # Get current active dimensions
        active_Tini = self.current_params['Tini']
        active_THorizon = self.current_params['THorizon']
        
        # Prepare matrices (pad if necessary)
        padded_matrices = self._prepare_matrices_for_solver(
            self.current_matrices, active_Tini, active_THorizon
        )
        
        # Pad input vectors to maximum dimensions
        uini_padded = np.zeros(self.u_dim * self.Tini_max)
        yini_padded = np.zeros(self.y_dim * self.Tini_max)
        yref_padded = np.zeros(self.y_dim * self.THorizon_max)
        
        # Copy active portions
        uini_len = min(len(uini), self.u_dim * active_Tini)
        yini_len = min(len(yini), self.y_dim * active_Tini)
        yref_len = min(len(yref), self.y_dim * active_THorizon)
        
        uini_padded[:uini_len] = uini.flatten()[:uini_len]
        yini_padded[:yini_len] = yini.flatten()[:yini_len]
        yref_padded[:yref_len] = yref.flatten()[:yref_len]
        
        # Pad cost matrices to maximum dimensions
        Q_padded = np.zeros((self.y_dim * self.THorizon_max, self.y_dim * self.THorizon_max))
        R_padded = np.zeros((self.u_dim * self.THorizon_max, self.u_dim * self.THorizon_max))
        
        Q_size = min(Q_val.shape[0], self.y_dim * active_THorizon)
        R_size = min(R_val.shape[0], self.u_dim * active_THorizon)
        
        Q_padded[:Q_size, :Q_size] = Q_val[:Q_size, :Q_size]
        R_padded[:R_size, :R_size] = R_val[:R_size, :R_size]
        
        # Pad regularization matrices
        lambda_y_padded = np.zeros((self.y_dim * self.Tini_max, self.y_dim * self.Tini_max))
        lambda_u_padded = np.zeros((self.u_dim * self.Tini_max, self.u_dim * self.Tini_max))
        lambda_g_padded = np.zeros((self.g_dim, self.g_dim))
        
        ly_size = min(lambda_y_val.shape[0], self.y_dim * active_Tini)
        lu_size = min(lambda_u_val.shape[0], self.u_dim * active_Tini)
        lg_size = min(lambda_g_val.shape[0], self.g_dim)
        
        lambda_y_padded[:ly_size, :ly_size] = lambda_y_val[:ly_size, :ly_size]
        lambda_u_padded[:lu_size, :lu_size] = lambda_u_val[:lu_size, :lu_size]
        lambda_g_padded[:lg_size, :lg_size] = lambda_g_val[:lg_size, :lg_size]
        
        # Construct parameter vector
        parameters = np.concatenate([
            uini_padded, yini_padded, yref_padded,
            padded_matrices['Up'].ravel(), padded_matrices['Yp'].ravel(),
            padded_matrices['Uf'].ravel(), padded_matrices['Yf'].ravel(),
            Q_padded.ravel(), R_padded.ravel(),
            lambda_y_padded.ravel(), lambda_g_padded.ravel(), lambda_u_padded.ravel(),
            [active_Tini], [active_THorizon]
        ])
        
        # Initial guess for g
        if g_prev is not None and len(g_prev) >= self.g_dim:
            g0 = g_prev[:self.g_dim]
        else:
            # Generate initial guess using least squares
            try:
                A_ls = np.vstack([padded_matrices['Up'][:active_Tini*self.u_dim, :self.g_dim],
                                padded_matrices['Yp'][:active_Tini*self.y_dim, :self.g_dim]])
                b_ls = np.vstack([uini_padded[:active_Tini*self.u_dim].reshape(-1,1),
                                yini_padded[:active_Tini*self.y_dim].reshape(-1,1)])
                g0 = np.linalg.lstsq(A_ls, b_ls, rcond=None)[0].flatten()
                if len(g0) < self.g_dim:
                    g0 = np.pad(g0, (0, self.g_dim - len(g0)), 'constant')
                else:
                    g0 = g0[:self.g_dim]
            except:
                g0 = np.zeros(self.g_dim)
        
        # Set solver parameters
        self.solver.set(0, "x", g0)
        self.solver.set(0, "p", parameters)
        
        # Solve
        t0 = time.time()
        status = self.solver.solve()
        solve_time = (time.time() - t0) * 1000  # Convert to ms
        
        # Extract results
        feasible = (status == 0)
        g_opt = self.solver.get(0, "x")
        cost = self.solver.get_cost() if feasible else float('inf')
        
        # Compute optimal control sequence
        u_opt = padded_matrices['Uf'][:active_THorizon*self.u_dim, :self.g_dim] @ g_opt
        
        return u_opt, g_opt, solve_time, feasible, cost
    
    def get_current_matrices_info(self):
        """Get information about currently active matrices."""
        if self.current_matrices is None:
            return {"status": "No active matrices"}
        
        return {
            "status": "Active",
            "method": self.current_matrices.get('method', 'unknown'),
            "source_speed": self.current_matrices.get('source_speed', 'unknown'),
            "target_speed": self.current_matrices.get('target_speed', self.last_speed),
            "params": self.current_params,
            "extrapolation": self.current_matrices.get('extrapolation', 0.0),
            "condition_numbers": self.current_matrices.get('condition_numbers', {})
        }


# Factory function for backward compatibility
def create_fixed_deepc_solver(hankel_data_file, ineqconidx=None, ineqconbd=None):
    """
    Factory function to create fixed DeePC solver.
    
    Args:
        hankel_data_file: Path to pre-computed Hankel matrices
        ineqconidx: Inequality constraint indices
        ineqconbd: Inequality constraint bounds
        
    Returns:
        DeePCFixedHankelSolver: Initialized solver
    """
    solver = DeePCFixedHankelSolver(
        u_dim=1, y_dim=1,
        hankel_data_file=hankel_data_file,
        ineqconidx=ineqconidx,
        ineqconbd=ineqconbd
    )
    return solver


def main():
    """Test the fixed Hankel DeePC solver."""
    from pathlib import Path
    
    # Find latest Hankel matrix file
    hankel_dir = Path("dataForHankle/OptimizedMatrices")
    pickle_files = list(hankel_dir.glob("complete_hankel_collection_*.pkl"))
    
    if not pickle_files:
        print("No Hankel matrix files found. Run HankelMatrixAnalyzer first.")
        return
    
    latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
    print(f"Testing with: {latest_file}")
    
    # Define constraints
    ineqconidx = {'u': [0], 'y': [0]}
    ineqconbd = {
        'lbu': np.array([-100.0]),
        'ubu': np.array([100.0]),
        'lby': np.array([0.0]),
        'uby': np.array([150.0])
    }
    
    # Create solver
    solver = DeePCFixedHankelSolver(
        hankel_data_file=str(latest_file),
        ineqconidx=ineqconidx,
        ineqconbd=ineqconbd
    )
    
    # Initialize Acados solver
    solver.init_acados_solver(recompile_solver=True)
    
    # Test solve at different speeds
    test_speeds = [20, 40, 60, 80]
    
    for speed in test_speeds:
        print(f"\nTesting at {speed} kph...")
        
        # Mock problem data
        Tini = 25
        THorizon = 15
        
        uini = np.random.randn(Tini, 1) * 10
        yini = np.ones((Tini, 1)) * speed + np.random.randn(Tini, 1) * 2
        yref = np.ones((THorizon, 1)) * speed
        
        Q_val = np.eye(THorizon) * 100
        R_val = np.eye(THorizon) * 0.1
        lambda_g_val = np.eye(solver.g_dim) * 10
        lambda_y_val = np.eye(Tini) * 20
        lambda_u_val = np.eye(Tini) * 5
        
        try:
            u_opt, g_opt, solve_time, feasible, cost = solver.solve_step(
                uini, yini, yref, speed,
                Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val
            )
            
            print(f"  Solve time: {solve_time:.2f} ms")
            print(f"  Feasible: {feasible}")
            print(f"  Cost: {cost:.2f}")
            print(f"  Control: {u_opt[0]:.2f}%")
            
            # Print matrix info
            info = solver.get_current_matrices_info()
            print(f"  Matrix method: {info['method']}")
            print(f"  Source speed: {info['source_speed']}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nFixed Hankel DeePC solver test completed!")


if __name__ == "__main__":
    main()