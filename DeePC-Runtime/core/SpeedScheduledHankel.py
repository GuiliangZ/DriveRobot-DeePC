#!/usr/bin/env python3
"""
Speed-Scheduled Hankel Matrix System for DeePC Control

Implements intelligent selection and interpolation of Hankel matrices based on
current vehicle speed for optimal DeePC performance across operating ranges.

Author: Generated for Drive Robot DeePC Project  
Date: 2025-07-21
"""

import numpy as np
import pickle
from pathlib import Path
from scipy import interpolate
from scipy.spatial.distance import cdist
import json
from datetime import datetime
import warnings


class SpeedScheduledHankel:
    """
    Speed-scheduled Hankel matrix system that selects optimal matrices
    based on current operating conditions.
    """
    
    def __init__(self, hankel_data_file=None):
        self.hankel_matrices = {}
        self.speed_points = []
        self.interpolators = {}
        self.current_matrices = None
        self.current_speed = None
        self.last_update_speed = None
        
        # Configuration parameters
        self.config = {
            'interpolation_method': 'linear',  # 'linear', 'nearest', 'rbf'
            'update_threshold': 2.0,           # Speed change threshold for matrix update (kph)
            'extrapolation_limit': 10.0,      # Max speed extrapolation beyond data (kph)
            'matrix_cache_size': 5,            # Number of matrices to cache
            'blend_window': 5.0,              # Speed window for matrix blending (kph)
            'use_matrix_blending': True,      # Enable smooth matrix transitions
            'min_g_dim': 10                   # Minimum degrees of freedom
        }
        
        if hankel_data_file:
            self.load_hankel_matrices(hankel_data_file)
    
    def load_hankel_matrices(self, hankel_data_file):
        """
        Load pre-computed Hankel matrices from file.
        
        Args:
            hankel_data_file: Path to pickle file with Hankel matrices
        """
        try:
            with open(hankel_data_file, 'rb') as f:
                self.hankel_matrices = pickle.load(f)
            
            self.speed_points = sorted(self.hankel_matrices.keys())
            
            print(f"[SpeedScheduled] Loaded Hankel matrices for {len(self.speed_points)} operating points:")
            print(f"  Speed range: {min(self.speed_points):.0f} - {max(self.speed_points):.0f} kph")
            
            # Validate matrices and extract key parameters
            self._validate_matrices()
            self._setup_interpolators()
            
            return True
            
        except Exception as e:
            print(f"[SpeedScheduled] Failed to load Hankel matrices: {e}")
            return False
    
    def _validate_matrices(self):
        """Validate loaded Hankel matrices and extract metadata."""
        print("[SpeedScheduled] Validating matrix collection...")
        
        valid_speeds = []
        matrix_info = {}
        
        for speed in self.speed_points:
            try:
                data = self.hankel_matrices[speed]
                
                # Check required fields
                required_fields = ['Up', 'Uf', 'Yp', 'Yf', 'params']
                if not all(field in data for field in required_fields):
                    print(f"  Warning: Missing fields for speed {speed} kph")
                    continue
                
                # Check matrix dimensions
                Up, Uf, Yp, Yf = data['Up'], data['Uf'], data['Yp'], data['Yf']
                params = data['params']
                
                # Validate SISO system dimensions
                if Up.shape[0] != params['Tini'] or Yp.shape[0] != params['Tini']:
                    print(f"  Warning: Dimension mismatch for speed {speed} kph")
                    continue
                
                # Check degrees of freedom
                if params['g_dim'] < self.config['min_g_dim']:
                    print(f"  Warning: Low g_dim ({params['g_dim']}) for speed {speed} kph")
                    continue
                
                # Store matrix metadata
                matrix_info[speed] = {
                    'Tini': params['Tini'],
                    'THorizon': params['THorizon'],
                    'g_dim': params['g_dim'],
                    'matrix_shapes': {
                        'Up': Up.shape, 'Uf': Uf.shape,
                        'Yp': Yp.shape, 'Yf': Yf.shape
                    },
                    'condition_numbers': {
                        'Up': np.linalg.cond(Up),
                        'Yp': np.linalg.cond(Yp)
                    },
                    'quality_score': data.get('quality_metrics', {}).get('quality_score', 0.0)
                }
                
                valid_speeds.append(speed)
                
            except Exception as e:
                print(f"  Error validating speed {speed} kph: {e}")
                continue
        
        # Update with only valid matrices
        self.speed_points = sorted(valid_speeds)
        self.matrix_info = matrix_info
        
        print(f"[SpeedScheduled] Validation complete: {len(self.speed_points)} valid matrices")
        
        if len(self.speed_points) < 2:
            raise ValueError("Need at least 2 valid speed points for interpolation")
    
    def _setup_interpolators(self):
        """Setup interpolation functions for matrix parameters."""
        if len(self.speed_points) < 2:
            print("[SpeedScheduled] Not enough points for interpolation setup")
            return
        
        print("[SpeedScheduled] Setting up parameter interpolators...")
        
        speeds = np.array(self.speed_points)
        
        # Extract parameter arrays
        tini_values = np.array([self.matrix_info[s]['Tini'] for s in self.speed_points])
        thorizon_values = np.array([self.matrix_info[s]['THorizon'] for s in self.speed_points])
        gdim_values = np.array([self.matrix_info[s]['g_dim'] for s in self.speed_points])
        
        # Create interpolators for parameters (for consistency checking)
        try:
            self.interpolators = {
                'Tini': interpolate.interp1d(speeds, tini_values, kind='nearest', 
                                           bounds_error=False, fill_value='extrapolate'),
                'THorizon': interpolate.interp1d(speeds, thorizon_values, kind='nearest',
                                               bounds_error=False, fill_value='extrapolate'),
                'g_dim': interpolate.interp1d(speeds, gdim_values, kind='linear',
                                            bounds_error=False, fill_value='extrapolate')
            }
            print("[SpeedScheduled] Interpolators setup complete")
        except Exception as e:
            print(f"[SpeedScheduled] Interpolator setup failed: {e}")
    
    def get_matrices_for_speed(self, current_speed, force_update=False):
        """
        Get appropriate Hankel matrices for current speed.
        
        Args:
            current_speed: Current vehicle speed (kph)
            force_update: Force matrix update even if within threshold
            
        Returns:
            dict: Selected Hankel matrices and metadata
        """
        # Check if update is needed
        if not force_update and self.current_matrices is not None:
            if self.last_update_speed is not None:
                speed_change = abs(current_speed - self.last_update_speed)
                if speed_change < self.config['update_threshold']:
                    return self.current_matrices
        
        # Select method based on configuration and data availability
        if self.config['use_matrix_blending'] and len(self.speed_points) >= 3:
            matrices = self._get_blended_matrices(current_speed)
        else:
            matrices = self._get_nearest_matrices(current_speed)
        
        # Update cache
        self.current_matrices = matrices
        self.current_speed = current_speed
        self.last_update_speed = current_speed
        
        return matrices
    
    def _get_nearest_matrices(self, target_speed):
        """Get matrices from nearest operating point."""
        # Find nearest speed point
        distances = [abs(target_speed - speed) for speed in self.speed_points]
        nearest_idx = np.argmin(distances)
        nearest_speed = self.speed_points[nearest_idx]
        
        # Get matrices for nearest speed
        data = self.hankel_matrices[nearest_speed]
        
        result = {
            'Up': data['Up'].copy(),
            'Uf': data['Uf'].copy(), 
            'Yp': data['Yp'].copy(),
            'Yf': data['Yf'].copy(),
            'params': data['params'].copy(),
            'method': 'nearest',
            'source_speed': nearest_speed,
            'target_speed': target_speed,
            'speed_error': abs(target_speed - nearest_speed),
            'extrapolation': self._check_extrapolation(target_speed)
        }
        
        return result
    
    def _get_blended_matrices(self, target_speed):
        """Get matrices using weighted blending of nearby operating points."""
        # Find surrounding speed points for interpolation
        if target_speed <= min(self.speed_points):
            # Use lowest speed point
            return self._get_nearest_matrices(target_speed)
        elif target_speed >= max(self.speed_points):
            # Use highest speed point  
            return self._get_nearest_matrices(target_speed)
        else:
            # Find bracketing speeds for interpolation
            lower_speeds = [s for s in self.speed_points if s <= target_speed]
            upper_speeds = [s for s in self.speed_points if s > target_speed]
            
            if not lower_speeds or not upper_speeds:
                return self._get_nearest_matrices(target_speed)
            
            speed_low = max(lower_speeds)
            speed_high = min(upper_speeds)
            
            # Check if we need blending based on distance
            blend_distance = min(
                abs(target_speed - speed_low),
                abs(target_speed - speed_high)
            )
            
            if blend_distance > self.config['blend_window']:
                # Too far from any point, use nearest
                return self._get_nearest_matrices(target_speed)
            
            # Perform matrix interpolation/blending
            return self._interpolate_matrices(target_speed, speed_low, speed_high)
    
    def _interpolate_matrices(self, target_speed, speed_low, speed_high):
        """
        Interpolate between two sets of Hankel matrices.
        
        Args:
            target_speed: Target interpolation speed
            speed_low: Lower speed point
            speed_high: Upper speed point
            
        Returns:
            dict: Interpolated matrices
        """
        # Calculate interpolation weight
        weight = (target_speed - speed_low) / (speed_high - speed_low)
        weight = np.clip(weight, 0.0, 1.0)
        
        data_low = self.hankel_matrices[speed_low]
        data_high = self.hankel_matrices[speed_high]
        
        # Check if matrices have compatible dimensions
        if (data_low['params']['Tini'] != data_high['params']['Tini'] or
            data_low['params']['THorizon'] != data_high['params']['THorizon']):
            # Different dimensions - use nearest instead
            print(f"[SpeedScheduled] Dimension mismatch between {speed_low} and {speed_high} kph, using nearest")
            return self._get_nearest_matrices(target_speed)
        
        try:
            # Linear interpolation of matrices
            Up_interp = (1 - weight) * data_low['Up'] + weight * data_high['Up']
            Uf_interp = (1 - weight) * data_low['Uf'] + weight * data_high['Uf']
            Yp_interp = (1 - weight) * data_low['Yp'] + weight * data_high['Yp']
            Yf_interp = (1 - weight) * data_low['Yf'] + weight * data_high['Yf']
            
            # Check interpolated matrix quality
            cond_Up = np.linalg.cond(Up_interp)
            cond_Yp = np.linalg.cond(Yp_interp)
            
            if max(cond_Up, cond_Yp) > 1e12:
                print(f"[SpeedScheduled] Poor conditioning after interpolation, using nearest")
                return self._get_nearest_matrices(target_speed)
            
            result = {
                'Up': Up_interp,
                'Uf': Uf_interp,
                'Yp': Yp_interp, 
                'Yf': Yf_interp,
                'params': data_low['params'].copy(),  # Use consistent parameters
                'method': 'interpolated',
                'source_speeds': [speed_low, speed_high],
                'interpolation_weight': weight,
                'target_speed': target_speed,
                'condition_numbers': {'Up': cond_Up, 'Yp': cond_Yp},
                'extrapolation': False
            }
            
            return result
            
        except Exception as e:
            print(f"[SpeedScheduled] Interpolation failed: {e}, using nearest")
            return self._get_nearest_matrices(target_speed)
    
    def _check_extrapolation(self, target_speed):
        """Check if target speed requires extrapolation beyond data range."""
        min_speed = min(self.speed_points)
        max_speed = max(self.speed_points)
        
        if target_speed < min_speed:
            return target_speed - min_speed  # Negative extrapolation
        elif target_speed > max_speed:
            return target_speed - max_speed  # Positive extrapolation
        else:
            return 0.0  # No extrapolation
    
    def get_system_info(self):
        """Get information about the loaded Hankel matrix system."""
        if not self.speed_points:
            return {"status": "No matrices loaded"}
        
        info = {
            "status": "Ready",
            "speed_range": [min(self.speed_points), max(self.speed_points)],
            "num_operating_points": len(self.speed_points),
            "operating_points": self.speed_points,
            "configuration": self.config,
            "parameter_ranges": {},
            "quality_summary": {}
        }
        
        if hasattr(self, 'matrix_info'):
            # Parameter ranges
            tini_values = [self.matrix_info[s]['Tini'] for s in self.speed_points]
            thorizon_values = [self.matrix_info[s]['THorizon'] for s in self.speed_points]
            gdim_values = [self.matrix_info[s]['g_dim'] for s in self.speed_points]
            
            info["parameter_ranges"] = {
                "Tini": [min(tini_values), max(tini_values)],
                "THorizon": [min(thorizon_values), max(thorizon_values)], 
                "g_dim": [min(gdim_values), max(gdim_values)]
            }
            
            # Quality summary
            quality_scores = [self.matrix_info[s]['quality_score'] for s in self.speed_points]
            condition_numbers = [max(self.matrix_info[s]['condition_numbers'].values()) for s in self.speed_points]
            
            info["quality_summary"] = {
                "avg_quality_score": np.mean(quality_scores),
                "min_quality_score": min(quality_scores),
                "max_condition_number": max(condition_numbers),
                "low_quality_points": [s for s in self.speed_points if self.matrix_info[s]['quality_score'] < 0.7]
            }
        
        return info
    
    def test_speed_range(self, test_speeds=None, verbose=True):
        """
        Test matrix selection across a range of speeds.
        
        Args:
            test_speeds: List of speeds to test (default: 0-120 kph in 5 kph increments)
            verbose: Print detailed results
            
        Returns:
            dict: Test results
        """
        if test_speeds is None:
            test_speeds = list(range(0, 121, 5))  # 0-120 kph in 5 kph steps
        
        results = {
            'test_speeds': test_speeds,
            'results': [],
            'statistics': {}
        }
        
        if verbose:
            print(f"\n[SpeedScheduled] Testing matrix selection for {len(test_speeds)} speeds...")
            print("Speed | Method      | Source     | Error   | Condition | Extrapolation")
            print("-" * 70)
        
        for speed in test_speeds:
            try:
                matrices = self.get_matrices_for_speed(speed, force_update=True)
                
                # Calculate metrics
                condition_up = np.linalg.cond(matrices['Up']) if 'Up' in matrices else np.inf
                condition_yp = np.linalg.cond(matrices['Yp']) if 'Yp' in matrices else np.inf
                max_condition = max(condition_up, condition_yp)
                
                test_result = {
                    'speed': speed,
                    'method': matrices.get('method', 'unknown'),
                    'source_speed': matrices.get('source_speed', matrices.get('source_speeds', None)),
                    'speed_error': matrices.get('speed_error', 0.0),
                    'max_condition_number': max_condition,
                    'extrapolation': matrices.get('extrapolation', 0.0),
                    'success': True
                }
                
                if verbose:
                    source_str = str(matrices.get('source_speed', matrices.get('source_speeds', 'N/A')))
                    if len(source_str) > 10:
                        source_str = source_str[:10] + "..."
                    
                    print(f"{speed:5.0f} | {matrices.get('method', 'unknown'):<11} | "
                          f"{source_str:<10} | {test_result['speed_error']:7.1f} | "
                          f"{max_condition:9.1e} | {test_result['extrapolation']:6.1f}")
                
            except Exception as e:
                test_result = {
                    'speed': speed,
                    'success': False,
                    'error': str(e)
                }
                
                if verbose:
                    print(f"{speed:5.0f} | ERROR       | {str(e)[:30]}")
            
            results['results'].append(test_result)
        
        # Calculate statistics
        successful_tests = [r for r in results['results'] if r.get('success', False)]
        
        if successful_tests:
            speed_errors = [r['speed_error'] for r in successful_tests if 'speed_error' in r]
            condition_numbers = [r['max_condition_number'] for r in successful_tests if 'max_condition_number' in r]
            extrapolations = [abs(r['extrapolation']) for r in successful_tests if 'extrapolation' in r]
            
            results['statistics'] = {
                'success_rate': len(successful_tests) / len(test_speeds),
                'avg_speed_error': np.mean(speed_errors) if speed_errors else 0,
                'max_speed_error': max(speed_errors) if speed_errors else 0,
                'avg_condition_number': np.mean(condition_numbers) if condition_numbers else 0,
                'max_condition_number': max(condition_numbers) if condition_numbers else 0,
                'max_extrapolation': max(extrapolations) if extrapolations else 0,
                'method_distribution': {}
            }
            
            # Method distribution
            methods = [r.get('method', 'unknown') for r in successful_tests]
            for method in set(methods):
                results['statistics']['method_distribution'][method] = methods.count(method)
        
        if verbose:
            print("-" * 70)
            stats = results['statistics']
            print(f"Success Rate: {stats.get('success_rate', 0)*100:.1f}%")
            print(f"Avg Speed Error: {stats.get('avg_speed_error', 0):.1f} kph")
            print(f"Max Condition Number: {stats.get('max_condition_number', 0):.1e}")
            print(f"Max Extrapolation: {stats.get('max_extrapolation', 0):.1f} kph")
        
        return results
    
    def save_configuration(self, filename):
        """Save current configuration to file."""
        config_data = {
            'speed_points': self.speed_points,
            'config': self.config,
            'system_info': self.get_system_info(),
            'created': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        print(f"[SpeedScheduled] Configuration saved to {filename}")


def main():
    """Test the speed-scheduled Hankel system."""
    # Find latest Hankel matrix file
    hankel_dir = Path("dataForHankle/OptimizedMatrices")
    
    if not hankel_dir.exists():
        print("No optimized Hankel matrices found. Run HankelMatrixAnalyzer first.")
        return
    
    # Find latest complete collection
    pickle_files = list(hankel_dir.glob("complete_hankel_collection_*.pkl"))
    if not pickle_files:
        print("No complete Hankel collection found.")
        return
    
    latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
    print(f"Using Hankel matrices from: {latest_file}")
    
    # Initialize speed-scheduled system
    scheduler = SpeedScheduledHankel(latest_file)
    
    # Print system information
    info = scheduler.get_system_info()
    print("\nSystem Information:")
    print(f"  Status: {info['status']}")
    print(f"  Operating Points: {info['num_operating_points']}")
    print(f"  Speed Range: {info['speed_range'][0]:.0f} - {info['speed_range'][1]:.0f} kph")
    print(f"  Parameter Ranges:")
    print(f"    Tini: {info['parameter_ranges']['Tini'][0]} - {info['parameter_ranges']['Tini'][1]}")
    print(f"    THorizon: {info['parameter_ranges']['THorizon'][0]} - {info['parameter_ranges']['THorizon'][1]}")
    print(f"    g_dim: {info['parameter_ranges']['g_dim'][0]} - {info['parameter_ranges']['g_dim'][1]}")
    print(f"  Quality Summary:")
    print(f"    Avg Quality Score: {info['quality_summary']['avg_quality_score']:.3f}")
    print(f"    Max Condition Number: {info['quality_summary']['max_condition_number']:.1e}")
    
    # Test speed range
    test_results = scheduler.test_speed_range()
    
    # Save configuration
    config_file = hankel_dir / f"speed_scheduled_config_{datetime.now().strftime('%H%M_%m%d')}.json"
    scheduler.save_configuration(config_file)
    
    print(f"\nSpeed-scheduled Hankel system ready for integration!")


if __name__ == "__main__":
    main()