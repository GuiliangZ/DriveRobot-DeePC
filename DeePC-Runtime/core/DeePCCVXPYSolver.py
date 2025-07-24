#!/usr/bin/env python3
"""
DeePCCVXPYSolver.py
Author: Claude AI Assistant  
Date: 2025-01-21
Version: 1.0.0
Description: CVXPY-based DeePC solver as alternative to Acados for 10Hz operation
"""

import time
import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
from typing import Tuple, Optional, Dict, Any
import warnings

class DeePCCVXPYSolver:
    """
    CVXPY-based implementation of DeePC controller
    Alternative to Acados for cases where compilation issues occur
    """
    
    def __init__(self, u_dim: int, y_dim: int, T: int, Tini: int, Np: int,
                 ineqconidx: Optional[Dict] = None, ineqconbd: Optional[Dict] = None):
        """
        Initialize CVXPY-based DeePC solver
        
        Args:
            u_dim: Control input dimension (1 for SISO)
            y_dim: Output dimension (1 for SISO) 
            T: Length of offline collected data
            Tini: Initialization length
            Np: Prediction horizon length
            ineqconidx: Inequality constraint indices
            ineqconbd: Inequality constraint bounds
        """
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.T = T
        self.Tini = Tini
        self.Np = Np
        self.g_dim = T - Tini - Np + 1
        
        # Store constraint information
        self.ineqconidx = ineqconidx
        self.ineqconbd = ineqconbd
        
        # Initialize CVXPY problem variables (will be defined when problem is built)
        self.g = None
        self.prob = None
        self.params = {}
        
        # Performance tracking
        self.solve_times = []
        self.solve_status = []
        
    def build_problem(self) -> None:
        """
        Build the CVXPY optimization problem structure
        This only needs to be called once, then parameters can be updated
        """
        print(">> Building CVXPY DeePC optimization problem...")
        print(f"   Problem dimensions: u_dim={self.u_dim}, y_dim={self.y_dim}")
        print(f"   Horizons: Tini={self.Tini}, Np={self.Np}")
        print(f"   Decision variables: g_dim={self.g_dim}")
        
        # Validate problem dimensions
        if self.g_dim <= (self.u_dim + self.y_dim) * self.Tini:
            raise ValueError(f'Problem does not have enough degrees of freedom! '
                           f'g_dim={self.g_dim} <= {(self.u_dim + self.y_dim) * self.Tini}')
        
        # Define optimization variable as column vector for proper matrix operations
        self.g = cp.Variable((self.g_dim, 1), name='g')
        
        # Define parameters (these will be updated at each solve)
        self.params = {
            'uini': cp.Parameter((self.u_dim * self.Tini, 1), name='uini'),
            'yini': cp.Parameter((self.y_dim * self.Tini, 1), name='yini'), 
            'yref': cp.Parameter((self.y_dim * self.Np, 1), name='yref'),
            'Up': cp.Parameter((self.u_dim * self.Tini, self.g_dim), name='Up'),
            'Yp': cp.Parameter((self.y_dim * self.Tini, self.g_dim), name='Yp'),
            'Uf': cp.Parameter((self.u_dim * self.Np, self.g_dim), name='Uf'),
            'Yf': cp.Parameter((self.y_dim * self.Np, self.g_dim), name='Yf'),
            'Q': cp.Parameter((self.y_dim * self.Np, self.y_dim * self.Np), 
                            PSD=True, name='Q'),
            'R': cp.Parameter((self.u_dim * self.Np, self.u_dim * self.Np), 
                            PSD=True, name='R'),
            'lambda_g': cp.Parameter((self.g_dim, self.g_dim), PSD=True, name='lambda_g'),
            'lambda_y': cp.Parameter((self.y_dim * self.Tini, self.y_dim * self.Tini), 
                                   PSD=True, name='lambda_y'),
            'lambda_u': cp.Parameter((self.u_dim * self.Tini, self.u_dim * self.Tini), 
                                   PSD=True, name='lambda_u')
        }
        
        # Define cost function components using compatible CVXPY formulations
        # Tracking error: ||Yf @ g - yref||_Q^2
        tracking_error = self.params['Yf'] @ self.g - self.params['yref']
        # CVXPY expects vectors for quad_form, ensure proper shape
        tracking_error_vec = cp.reshape(tracking_error, (-1,), order='F')
        tracking_cost = cp.quad_form(tracking_error_vec, self.params['Q'])
        
        # Control effort: ||Uf @ g||_R^2  
        control_effort = self.params['Uf'] @ self.g
        control_effort_vec = cp.reshape(control_effort, (-1,), order='F')
        control_cost = cp.quad_form(control_effort_vec, self.params['R'])
        
        # Regularization terms for robust DeePC
        output_slack = self.params['Yp'] @ self.g - self.params['yini']
        output_slack_vec = cp.reshape(output_slack, (-1,), order='F')
        output_slack_cost = cp.quad_form(output_slack_vec, self.params['lambda_y'])
        
        input_slack = self.params['Up'] @ self.g - self.params['uini']
        input_slack_vec = cp.reshape(input_slack, (-1,), order='F')  
        input_slack_cost = cp.quad_form(input_slack_vec, self.params['lambda_u'])
        
        # g regularization
        g_regularization = cp.quad_form(self.g, self.params['lambda_g'])
        
        # Total objective
        objective = cp.Minimize(
            tracking_cost + control_cost + 
            output_slack_cost + input_slack_cost + 
            g_regularization
        )
        
        # Define constraints
        constraints = []
        
        # Equality constraint: Up @ g = uini (hard constraint on input consistency)
        constraints.append(self.params['Up'] @ self.g == self.params['uini'])
        
        # Inequality constraints on u and y
        if self.ineqconidx is not None:
            constraints.extend(self._build_inequality_constraints())
        
        # Create problem
        self.prob = cp.Problem(objective, constraints)
        
        print(f">> CVXPY problem built: {self.g_dim} variables, "
              f"{len(constraints)} constraints")
    
    def _build_inequality_constraints(self) -> list:
        """Build inequality constraints based on ineqconidx and ineqconbd"""
        constraints = []
        
        for varname, idx in self.ineqconidx.items():
            if varname == 'u':
                # Control input constraints
                if 'lbu' in self.ineqconbd and 'ubu' in self.ineqconbd:
                    u_pred = self.params['Uf'] @ self.g
                    
                    # Apply constraints to specified indices over prediction horizon
                    for i in range(self.Np):
                        for j in idx:
                            var_idx = i * self.u_dim + j
                            constraints.append(
                                u_pred[var_idx] >= self.ineqconbd['lbu'][j]
                            )
                            constraints.append(
                                u_pred[var_idx] <= self.ineqconbd['ubu'][j]
                            )
                            
            elif varname == 'y':
                # Output constraints
                if 'lby' in self.ineqconbd and 'uby' in self.ineqconbd:
                    y_pred = self.params['Yf'] @ self.g
                    
                    for i in range(self.Np):
                        for j in idx:
                            var_idx = i * self.y_dim + j
                            constraints.append(
                                y_pred[var_idx] >= self.ineqconbd['lby'][j]
                            )
                            constraints.append(
                                y_pred[var_idx] <= self.ineqconbd['uby'][j]
                            )
                            
            elif varname == 'du':
                # Rate of change constraints
                if 'lbdu' in self.ineqconbd and 'ubdu' in self.ineqconbd:
                    u_pred = self.params['Uf'] @ self.g
                    
                    # Current input trajectory
                    u_current = u_pred
                    # Previous input (last input from uini + predicted inputs)
                    u_prev = cp.vstack([
                        self.params['uini'][-self.u_dim:],
                        u_current[:-self.u_dim]
                    ])
                    
                    du = u_current - u_prev
                    
                    for i in range(self.Np):
                        for j in idx:
                            var_idx = i * self.u_dim + j
                            constraints.append(
                                du[var_idx] >= self.ineqconbd['lbdu'][j]
                            )
                            constraints.append(
                                du[var_idx] <= self.ineqconbd['ubdu'][j]
                            )
        
        return constraints
    
    def solve_step(self, uini: np.ndarray, yini: np.ndarray, yref: np.ndarray,
                   Up_cur: np.ndarray, Uf_cur: np.ndarray, 
                   Yp_cur: np.ndarray, Yf_cur: np.ndarray,
                   Q_val: np.ndarray, R_val: np.ndarray,
                   lambda_g_val: np.ndarray, lambda_y_val: np.ndarray, 
                   lambda_u_val: np.ndarray,
                   solver: str = 'OSQP', warm_start: bool = True,
                   max_iter: int = 2000) -> Tuple[np.ndarray, np.ndarray, float, bool, float]:
        """
        Solve one step of the DeePC optimization problem
        
        Args:
            uini, yini: Initial input/output sequences
            yref: Reference output trajectory
            Up_cur, Uf_cur, Yp_cur, Yf_cur: Current Hankel matrices
            Q_val, R_val: Weighting matrices
            lambda_*: Regularization parameters
            solver: CVXPY solver to use ('OSQP', 'SCS', 'CLARABEL')
            warm_start: Enable warm starting
            max_iter: Maximum solver iterations
            
        Returns:
            u_opt: Optimal control sequence
            g_opt: Optimal g vector
            solve_time: Time taken to solve (ms)
            success: Whether solve was successful
            cost: Optimal cost value
        """
        if self.prob is None:
            raise RuntimeError("Problem not built! Call build_problem() first.")
        
        # Update parameter values
        try:
            self.params['uini'].value = uini.reshape(-1, 1)
            self.params['yini'].value = yini.reshape(-1, 1)
            self.params['yref'].value = yref.reshape(-1, 1)
            self.params['Up'].value = Up_cur
            self.params['Yp'].value = Yp_cur
            self.params['Uf'].value = Uf_cur
            self.params['Yf'].value = Yf_cur
            self.params['Q'].value = Q_val
            self.params['R'].value = R_val
            self.params['lambda_g'].value = lambda_g_val
            self.params['lambda_y'].value = lambda_y_val
            self.params['lambda_u'].value = lambda_u_val
            
        except Exception as e:
            print(f"Parameter update failed: {e}")
            return np.zeros(self.Np), np.zeros(self.g_dim), 0.0, False, float('inf')
        
        # Solve problem
        start_time = time.time()
        
        try:
            # Set solver options for real-time performance
            solver_opts = {
                'max_iter': max_iter,
                'eps_abs': 1e-4,
                'eps_rel': 1e-4,
                'verbose': False,
                'warm_start': warm_start
            }
            
            if solver == 'OSQP':
                solver_opts['adaptive_rho'] = True
                solver_opts['polish'] = True
                solver_opts['polish_refine_iter'] = 5
            elif solver == 'SCS':
                solver_opts['eps'] = 1e-4
                solver_opts['normalize'] = True
            
            # Solve
            self.prob.solve(solver=solver, **solver_opts)
            
            solve_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Check solution status
            success = self.prob.status in ['optimal', 'optimal_inaccurate']
            
            if success and self.g.value is not None:
                g_opt = self.g.value.flatten()
                u_opt = (Uf_cur @ self.g.value).flatten()
                cost = self.prob.value if self.prob.value is not None else 0.0
            else:
                print(f"Solver failed with status: {self.prob.status}")
                g_opt = np.zeros(self.g_dim)
                u_opt = np.zeros(self.Np)
                cost = float('inf')
                success = False
                
        except Exception as e:
            print(f"Solver error: {e}")
            solve_time = (time.time() - start_time) * 1000
            g_opt = np.zeros(self.g_dim)
            u_opt = np.zeros(self.Np)
            cost = float('inf')
            success = False
        
        # Track performance
        self.solve_times.append(solve_time)
        self.solve_status.append(success)
        
        return u_opt, g_opt, solve_time, success, cost
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get solver performance statistics"""
        if not self.solve_times:
            return {}
            
        return {
            'avg_solve_time_ms': np.mean(self.solve_times),
            'max_solve_time_ms': np.max(self.solve_times),
            'min_solve_time_ms': np.min(self.solve_times),
            'success_rate': np.mean(self.solve_status),
            'total_solves': len(self.solve_times)
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking"""
        self.solve_times.clear()
        self.solve_status.clear()

# Integration wrapper for compatibility with existing DeePCAcados interface
class DeePCCVXPYWrapper:
    """
    Wrapper class to make CVXPY solver compatible with existing DeePCAcados interface
    """
    
    def __init__(self, u_dim: int, y_dim: int, T: int, Tini: int, Np: int,
                 ineqconidx: Optional[Dict] = None, ineqconbd: Optional[Dict] = None):
        """Initialize wrapper with same interface as DeePCAcados"""
        self.solver = DeePCCVXPYSolver(u_dim, y_dim, T, Tini, Np, ineqconidx, ineqconbd)
        self.built = False
    
    def init_DeePCAcadosSolver(self, recompile_solver: bool = True, 
                              ineqconidx: Optional[Dict] = None, 
                              ineqconbd: Optional[Dict] = None):
        """Initialize solver (compatible with DeePCAcados interface)"""
        if recompile_solver or not self.built:
            self.solver.build_problem()
            self.built = True
            print(">> CVXPY solver initialized")
    
    def acados_solver_step(self, uini: np.ndarray, yini: np.ndarray, yref: np.ndarray,
                          Up_cur: np.ndarray, Uf_cur: np.ndarray,
                          Yp_cur: np.ndarray, Yf_cur: np.ndarray,
                          Q_val: np.ndarray, R_val: np.ndarray,
                          lambda_g_val: np.ndarray, lambda_y_val: np.ndarray,
                          lambda_u_val: np.ndarray, g_prev: Optional[np.ndarray] = None,
                          solver: str = 'OSQP') -> Tuple[np.ndarray, np.ndarray, float, bool, float]:
        """Solve step (compatible with DeePCAcados interface)"""
        return self.solver.solve_step(
            uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur,
            Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val,
            solver=solver, warm_start=(g_prev is not None)
        )

if __name__ == "__main__":
    # Test the CVXPY solver
    print("Testing DeePCCVXPYSolver...")
    
    # Problem dimensions
    u_dim, y_dim = 1, 1
    T, Tini, Np = 100, 20, 20
    g_dim = T - Tini - Np + 1
    
    # Create solver
    ineqconidx = {'u': [0], 'y': [0]}
    ineqconbd = {'lbu': [-30], 'ubu': [100], 'lby': [0], 'uby': [140]}
    
    solver = DeePCCVXPYWrapper(u_dim, y_dim, T, Tini, Np, ineqconidx, ineqconbd)
    solver.init_DeePCAcadosSolver()
    
    # Test solve
    uini = np.random.randn(Tini, 1)
    yini = np.random.randn(Tini, 1) * 50 + 60
    yref = np.ones((Np, 1)) * 80
    
    Up_cur = np.random.randn(Tini, g_dim)
    Yp_cur = np.random.randn(Tini, g_dim)
    Uf_cur = np.random.randn(Np, g_dim)
    Yf_cur = np.random.randn(Np, g_dim)
    
    Q_val = np.eye(Np) * 265
    R_val = np.eye(Np) * 0.15
    lambda_g_val = np.eye(g_dim) * 60
    lambda_y_val = np.eye(Tini) * 10
    lambda_u_val = np.eye(Tini) * 10
    
    # Test solve
    u_opt, g_opt, solve_time, success, cost = solver.acados_solver_step(
        uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur,
        Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val
    )
    
    print(f"Solve successful: {success}")
    print(f"Solve time: {solve_time:.2f} ms")
    print(f"First control: {u_opt[0]:.2f}")
    print(f"Cost: {cost:.2f}")