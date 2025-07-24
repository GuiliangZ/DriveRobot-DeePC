#!/usr/bin/env python3
"""
Hankel Matrix Analyzer and Optimizer for Speed-Scheduled DeePC Control

Analyzes PRBS collected data to generate optimal Hankel matrices for different
speed operating points, with tools for matrix quality assessment and optimization.

Author: Generated for Drive Robot DeePC Project
Date: 2025-07-21
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize_scalar
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import DeePC utilities
from utils_deepc import hankel_full


class HankelMatrixAnalyzer:
    """
    Analyzes collected PRBS data and generates optimal Hankel matrices
    for speed-scheduled DeePC control.
    """
    
    def __init__(self, data_dir="dataForHankle/PRBSCollection"):
        self.data_dir = Path(data_dir)
        self.collected_data = {}
        self.hankel_matrices = {}
        self.analysis_results = {}
        
        # Default parameters (can be optimized)
        self.default_params = {
            'Tini': 25,
            'THorizon': 15,
            'min_data_length': 2000,
            'data_quality_threshold': 0.7
        }
        
    def load_collected_data(self, json_file=None):
        """Load PRBS collected data from JSON file."""
        if json_file is None:
            # Find latest collection file
            json_files = list(self.data_dir.glob("prbs_collection_*.json"))
            if not json_files:
                raise FileNotFoundError(f"No PRBS collection files found in {self.data_dir}")
            json_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        print(f"[Analyzer] Loading data from {json_file}")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Convert back to numpy arrays and organize by speed
        for speed_str, data in raw_data.items():
            speed = float(speed_str)
            self.collected_data[speed] = {
                'time': np.array(data['time']),
                'target_speed': np.array(data['target_speed']),
                'measured_speed': np.array(data['measured_speed']),
                'total_control': np.array(data['total_control']),
                'prbs_excitation': np.array(data['prbs_excitation'])
            }
        
        print(f"[Analyzer] Loaded data for {len(self.collected_data)} operating points")
        return True
    
    def analyze_data_quality(self, speed):
        """
        Analyze quality of collected data for a specific operating point.
        
        Args:
            speed: Operating point speed (kph)
            
        Returns:
            dict: Quality metrics
        """
        if speed not in self.collected_data:
            raise ValueError(f"No data available for speed {speed} kph")
        
        data = self.collected_data[speed]
        u_data = data['total_control']
        y_data = data['measured_speed']
        
        metrics = {}
        
        # Basic statistics
        metrics['n_samples'] = len(u_data)
        metrics['duration'] = data['time'][-1] if len(data['time']) > 0 else 0
        metrics['sampling_rate'] = len(u_data) / metrics['duration'] if metrics['duration'] > 0 else 0
        
        # Input signal characteristics
        metrics['u_mean'] = np.mean(u_data)
        metrics['u_std'] = np.std(u_data)
        metrics['u_range'] = np.ptp(u_data)  # peak-to-peak
        metrics['u_rms'] = np.sqrt(np.mean(u_data**2))
        
        # Output signal characteristics
        metrics['y_mean'] = np.mean(y_data)
        metrics['y_std'] = np.std(y_data)
        metrics['y_range'] = np.ptp(y_data)
        metrics['y_target'] = speed
        metrics['y_tracking_error'] = np.mean(np.abs(y_data - speed))
        metrics['y_tracking_rmse'] = np.sqrt(np.mean((y_data - speed)**2))
        
        # Signal richness (frequency content)
        # Compute power spectral density
        from scipy.signal import welch
        f_u, psd_u = welch(u_data, fs=metrics['sampling_rate'], nperseg=min(1024, len(u_data)//4))
        f_y, psd_y = welch(y_data, fs=metrics['sampling_rate'], nperseg=min(1024, len(y_data)//4))
        
        # Signal richness metric: energy distribution across frequency bands
        total_power_u = np.sum(psd_u)
        low_freq_power = np.sum(psd_u[f_u <= 1.0])  # Below 1 Hz
        metrics['u_richness'] = 1.0 - (low_freq_power / total_power_u) if total_power_u > 0 else 0
        
        total_power_y = np.sum(psd_y)
        low_freq_power_y = np.sum(psd_y[f_y <= 1.0])  # Below 1 Hz
        metrics['y_richness'] = 1.0 - (low_freq_power_y / total_power_y) if total_power_y > 0 else 0
        
        # Data persistence check
        # Compute rank of data matrix for persistency of excitation
        n_lags = min(50, len(u_data)//10)
        hankel_test = np.array([u_data[i:i+n_lags] for i in range(len(u_data)-n_lags+1)])
        metrics['data_rank'] = np.linalg.matrix_rank(hankel_test)
        metrics['expected_rank'] = n_lags
        metrics['persistence_ratio'] = metrics['data_rank'] / metrics['expected_rank']
        
        # Overall quality score (0-1)
        quality_factors = [
            min(1.0, metrics['n_samples'] / self.default_params['min_data_length']),
            min(1.0, metrics['u_richness'] * 2),  # Weight signal richness
            min(1.0, metrics['persistence_ratio']),
            max(0.0, 1.0 - metrics['y_tracking_rmse'] / 10.0)  # Penalize poor tracking
        ]
        metrics['quality_score'] = np.mean(quality_factors)
        
        return metrics
    
    def optimize_hankel_dimensions(self, speed, Tini_range=(15, 35), THorizon_range=(10, 25)):
        """
        Optimize Hankel matrix dimensions for a specific operating point.
        
        Args:
            speed: Operating point speed (kph)
            Tini_range: Range of initialization lengths to test
            THorizon_range: Range of prediction horizons to test
            
        Returns:
            dict: Optimal parameters and metrics
        """
        if speed not in self.collected_data:
            raise ValueError(f"No data available for speed {speed} kph")
        
        data = self.collected_data[speed]
        u_data = data['total_control']
        y_data = data['measured_speed']
        
        print(f"[Optimizer] Optimizing Hankel dimensions for {speed} kph...")
        
        # Grid search over parameter space
        best_params = None
        best_score = -np.inf
        results = []
        
        for Tini in range(Tini_range[0], Tini_range[1] + 1, 2):
            for THorizon in range(THorizon_range[0], THorizon_range[1] + 1, 2):
                T = len(u_data)  # Use all available data
                g_dim = T - Tini - THorizon + 1
                
                # Skip if not enough degrees of freedom
                if g_dim <= (1 + 1) * Tini:  # SISO system
                    continue
                
                if T < Tini + THorizon:
                    continue
                
                try:
                    # Build Hankel matrices
                    Up, Uf, Yp, Yf = hankel_full(
                        u_data.reshape(-1, 1), 
                        y_data.reshape(-1, 1), 
                        Tini, THorizon
                    )
                    
                    # Evaluate matrix quality
                    score = self.evaluate_hankel_quality(Up, Uf, Yp, Yf, Tini, THorizon)
                    
                    results.append({
                        'Tini': Tini,
                        'THorizon': THorizon,
                        'g_dim': g_dim,
                        'score': score,
                        'Up_cond': np.linalg.cond(Up),
                        'Yp_cond': np.linalg.cond(Yp)
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'Tini': Tini,
                            'THorizon': THorizon,
                            'g_dim': g_dim,
                            'score': score
                        }
                        
                except Exception as e:
                    # Skip problematic parameter combinations
                    continue
        
        if best_params is None:
            # Fallback to default parameters
            print(f"[Optimizer] No valid parameters found, using defaults")
            best_params = {
                'Tini': self.default_params['Tini'],
                'THorizon': self.default_params['THorizon'],
                'g_dim': len(u_data) - self.default_params['Tini'] - self.default_params['THorizon'] + 1,
                'score': 0.0
            }
        
        print(f"[Optimizer] Best parameters for {speed} kph: Tini={best_params['Tini']}, THorizon={best_params['THorizon']}, score={best_params['score']:.3f}")
        
        return best_params, results
    
    def evaluate_hankel_quality(self, Up, Uf, Yp, Yf, Tini, THorizon):
        """
        Evaluate the quality of Hankel matrices for DeePC control.
        
        Args:
            Up, Uf, Yp, Yf: Hankel matrices
            Tini: Initialization length
            THorizon: Prediction horizon
            
        Returns:
            float: Quality score (higher is better)
        """
        score = 0.0
        
        try:
            # 1. Rank analysis - matrices should have good rank
            rank_Up = np.linalg.matrix_rank(Up)
            rank_Yp = np.linalg.matrix_rank(Yp)
            rank_score = (rank_Up + rank_Yp) / (Up.shape[0] + Yp.shape[0])
            score += 0.3 * rank_score
            
            # 2. Condition number - lower is better (more numerical stability)
            cond_Up = np.linalg.cond(Up)
            cond_Yp = np.linalg.cond(Yp)
            cond_score = 2.0 / (1.0 + np.log10(max(cond_Up, cond_Yp, 1.0)))
            score += 0.2 * cond_score
            
            # 3. SVD analysis - check for clear singular value separation
            s_Up = np.linalg.svd(Up, compute_uv=False)
            s_Yp = np.linalg.svd(Yp, compute_uv=False)
            
            # Look for gap in singular values (indicates clear system order)
            def sv_gap_score(s):
                if len(s) < 2:
                    return 0.0
                s_norm = s / s[0]  # Normalize by largest SV
                # Find largest gap
                gaps = s_norm[:-1] - s_norm[1:]
                return np.max(gaps) if len(gaps) > 0 else 0.0
            
            sv_score = (sv_gap_score(s_Up) + sv_gap_score(s_Yp)) / 2.0
            score += 0.2 * sv_score
            
            # 4. Data richness - how well does the data span the space
            data_matrix = np.vstack([Up, Yp])
            data_richness = np.linalg.matrix_rank(data_matrix) / min(data_matrix.shape)
            score += 0.15 * data_richness
            
            # 5. Dimension efficiency - prefer smaller matrices for computational efficiency
            total_params = Up.size + Uf.size + Yp.size + Yf.size
            efficiency_score = 1.0 / (1.0 + np.log10(total_params / 1000.0))
            score += 0.15 * efficiency_score
            
        except Exception as e:
            # Return low score if evaluation fails
            score = 0.0
        
        return score
    
    def generate_optimal_hankel_matrices(self):
        """
        Generate optimal Hankel matrices for all collected operating points.
        
        Returns:
            dict: Hankel matrices organized by operating point
        """
        print("\n[Generator] Generating optimal Hankel matrices...")
        
        for speed in sorted(self.collected_data.keys()):
            print(f"\n[Generator] Processing {speed} kph operating point...")
            
            # Analyze data quality
            quality_metrics = self.analyze_data_quality(speed)
            
            if quality_metrics['quality_score'] < self.default_params['data_quality_threshold']:
                print(f"[Generator] WARNING: Low data quality for {speed} kph (score: {quality_metrics['quality_score']:.3f})")
            
            # Optimize Hankel dimensions
            best_params, optimization_results = self.optimize_hankel_dimensions(speed)
            
            # Generate optimal Hankel matrices
            data = self.collected_data[speed]
            u_data = data['total_control'].reshape(-1, 1)
            y_data = data['measured_speed'].reshape(-1, 1)
            
            Up, Uf, Yp, Yf = hankel_full(u_data, y_data, best_params['Tini'], best_params['THorizon'])
            
            # Store results
            self.hankel_matrices[speed] = {
                'Up': Up,
                'Uf': Uf, 
                'Yp': Yp,
                'Yf': Yf,
                'params': best_params,
                'quality_metrics': quality_metrics,
                'optimization_results': optimization_results
            }
            
            print(f"[Generator] Generated matrices for {speed} kph:")
            print(f"  Tini: {best_params['Tini']}, THorizon: {best_params['THorizon']}")
            print(f"  Up: {Up.shape}, Uf: {Uf.shape}, Yp: {Yp.shape}, Yf: {Yf.shape}")
            print(f"  Quality score: {quality_metrics['quality_score']:.3f}")
            print(f"  Optimization score: {best_params['score']:.3f}")
        
        # Generate analysis summary
        self.analysis_results = {
            'speeds': list(self.hankel_matrices.keys()),
            'generation_time': datetime.now().isoformat(),
            'total_operating_points': len(self.hankel_matrices),
            'parameter_ranges': {
                'Tini': [self.hankel_matrices[s]['params']['Tini'] for s in self.hankel_matrices.keys()],
                'THorizon': [self.hankel_matrices[s]['params']['THorizon'] for s in self.hankel_matrices.keys()],
                'g_dim': [self.hankel_matrices[s]['params']['g_dim'] for s in self.hankel_matrices.keys()]
            },
            'quality_scores': [self.hankel_matrices[s]['quality_metrics']['quality_score'] for s in self.hankel_matrices.keys()]
        }
        
        print(f"\n[Generator] Generated Hankel matrices for {len(self.hankel_matrices)} operating points")
        return self.hankel_matrices
    
    def save_hankel_matrices(self, save_dir="dataForHankle/OptimizedMatrices"):
        """Save generated Hankel matrices to files."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%H%M_%m%d")
        
        # Save individual matrices as .npz files
        for speed, data in self.hankel_matrices.items():
            filename = save_path / f"hankel_{speed}kph_{timestamp}.npz"
            np.savez(
                filename,
                Up=data['Up'], Uf=data['Uf'], 
                Yp=data['Yp'], Yf=data['Yf'],
                Tini=data['params']['Tini'],
                THorizon=data['params']['THorizon'],
                g_dim=data['params']['g_dim'],
                quality_score=data['quality_metrics']['quality_score']
            )
        
        # Save complete collection as pickle file
        complete_file = save_path / f"complete_hankel_collection_{timestamp}.pkl"
        with open(complete_file, 'wb') as f:
            pickle.dump(self.hankel_matrices, f)
        
        # Save analysis summary as JSON
        summary_file = save_path / f"hankel_analysis_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"[Save] Hankel matrices saved to {save_path}")
        print(f"  Individual files: hankel_*kph_{timestamp}.npz")
        print(f"  Complete collection: {complete_file.name}")
        print(f"  Analysis summary: {summary_file.name}")
        
        return complete_file
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        if not self.hankel_matrices:
            print("[Report] No Hankel matrices available for analysis")
            return
        
        print("\n" + "="*80)
        print("HANKEL MATRIX ANALYSIS REPORT")
        print("="*80)
        
        speeds = sorted(self.hankel_matrices.keys())
        
        # Summary statistics
        print(f"\nSUMMARY:")
        print(f"  Operating Points: {len(speeds)}")
        print(f"  Speed Range: {min(speeds):.0f} - {max(speeds):.0f} kph")
        
        # Parameter analysis
        tini_values = [self.hankel_matrices[s]['params']['Tini'] for s in speeds]
        thorizon_values = [self.hankel_matrices[s]['params']['THorizon'] for s in speeds]
        gdim_values = [self.hankel_matrices[s]['params']['g_dim'] for s in speeds]
        
        print(f"\nPARAMETER RANGES:")
        print(f"  Tini: {min(tini_values)} - {max(tini_values)} (avg: {np.mean(tini_values):.1f})")
        print(f"  THorizon: {min(thorizon_values)} - {max(thorizon_values)} (avg: {np.mean(thorizon_values):.1f})")
        print(f"  g_dim: {min(gdim_values)} - {max(gdim_values)} (avg: {np.mean(gdim_values):.1f})")
        
        # Quality analysis
        quality_scores = [self.hankel_matrices[s]['quality_metrics']['quality_score'] for s in speeds]
        print(f"\nQUALITY METRICS:")
        print(f"  Average Quality Score: {np.mean(quality_scores):.3f}")
        print(f"  Quality Range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
        print(f"  Low Quality Points (<0.7): {sum(1 for q in quality_scores if q < 0.7)}")
        
        # Detailed breakdown by operating point
        print(f"\nDETAILED BREAKDOWN:")
        print("Speed | Tini | THor | g_dim | Quality | OptScore | Up_cond | Yp_cond")
        print("-" * 70)
        
        for speed in speeds:
            data = self.hankel_matrices[speed]
            params = data['params']
            quality = data['quality_metrics']
            
            Up_cond = np.linalg.cond(data['Up'])
            Yp_cond = np.linalg.cond(data['Yp'])
            
            print(f"{speed:5.0f} | {params['Tini']:4d} | {params['THorizon']:4d} | "
                  f"{params['g_dim']:5d} | {quality['quality_score']:7.3f} | "
                  f"{params['score']:8.3f} | {Up_cond:7.1e} | {Yp_cond:7.1e}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        
        low_quality_speeds = [s for s in speeds if self.hankel_matrices[s]['quality_metrics']['quality_score'] < 0.7]
        if low_quality_speeds:
            print(f"  ⚠ Consider recollecting data for speeds: {low_quality_speeds}")
        
        high_condition_speeds = []
        for speed in speeds:
            data = self.hankel_matrices[speed]
            Up_cond = np.linalg.cond(data['Up'])
            Yp_cond = np.linalg.cond(data['Yp'])
            if max(Up_cond, Yp_cond) > 1e12:
                high_condition_speeds.append(speed)
        
        if high_condition_speeds:
            print(f"  ⚠ High condition numbers at speeds: {high_condition_speeds}")
        
        if not low_quality_speeds and not high_condition_speeds:
            print(f"  ✓ All matrices meet quality requirements")
        
        print("="*80)
    
    def plot_analysis_results(self, save_plots=True, save_dir="dataForHankle/OptimizedMatrices"):
        """Generate plots for analysis results."""
        if not self.hankel_matrices:
            print("[Plot] No data available for plotting")
            return
        
        speeds = sorted(self.hankel_matrices.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hankel Matrix Analysis Results', fontsize=16)
        
        # 1. Parameter values vs speed
        tini_values = [self.hankel_matrices[s]['params']['Tini'] for s in speeds]
        thorizon_values = [self.hankel_matrices[s]['params']['THorizon'] for s in speeds]
        gdim_values = [self.hankel_matrices[s]['params']['g_dim'] for s in speeds]
        
        axes[0,0].plot(speeds, tini_values, 'o-', label='Tini')
        axes[0,0].plot(speeds, thorizon_values, 's-', label='THorizon')
        axes[0,0].set_xlabel('Speed (kph)')
        axes[0,0].set_ylabel('Parameter Value')
        axes[0,0].set_title('Optimal Parameters vs Speed')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. g_dim vs speed
        axes[0,1].plot(speeds, gdim_values, 'o-', color='green')
        axes[0,1].set_xlabel('Speed (kph)')
        axes[0,1].set_ylabel('g_dim (degrees of freedom)')
        axes[0,1].set_title('Degrees of Freedom vs Speed')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Quality scores vs speed
        quality_scores = [self.hankel_matrices[s]['quality_metrics']['quality_score'] for s in speeds]
        opt_scores = [self.hankel_matrices[s]['params']['score'] for s in speeds]
        
        axes[0,2].plot(speeds, quality_scores, 'o-', label='Data Quality', color='blue')
        axes[0,2].plot(speeds, opt_scores, 's-', label='Optimization Score', color='red')
        axes[0,2].axhline(y=0.7, color='orange', linestyle='--', label='Quality Threshold')
        axes[0,2].set_xlabel('Speed (kph)')
        axes[0,2].set_ylabel('Score')
        axes[0,2].set_title('Quality Metrics vs Speed')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Condition numbers vs speed
        up_conds = [np.log10(np.linalg.cond(self.hankel_matrices[s]['Up'])) for s in speeds]
        yp_conds = [np.log10(np.linalg.cond(self.hankel_matrices[s]['Yp'])) for s in speeds]
        
        axes[1,0].semilogy(speeds, [10**c for c in up_conds], 'o-', label='Up condition')
        axes[1,0].semilogy(speeds, [10**c for c in yp_conds], 's-', label='Yp condition') 
        axes[1,0].set_xlabel('Speed (kph)')
        axes[1,0].set_ylabel('Condition Number')
        axes[1,0].set_title('Matrix Condition Numbers')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Matrix sizes vs speed
        up_sizes = [self.hankel_matrices[s]['Up'].size for s in speeds]
        total_sizes = [sum([self.hankel_matrices[s][m].size for m in ['Up','Uf','Yp','Yf']]) for s in speeds]
        
        axes[1,1].plot(speeds, up_sizes, 'o-', label='Up matrix size')
        axes[1,1].plot(speeds, total_sizes, 's-', label='Total matrix size')
        axes[1,1].set_xlabel('Speed (kph)')
        axes[1,1].set_ylabel('Matrix Elements')
        axes[1,1].set_title('Matrix Sizes vs Speed')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Data quality breakdown
        tracking_errors = [self.hankel_matrices[s]['quality_metrics']['y_tracking_rmse'] for s in speeds]
        richness_scores = [self.hankel_matrices[s]['quality_metrics']['u_richness'] for s in speeds]
        
        ax1 = axes[1,2]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(speeds, tracking_errors, 'o-', color='red', label='Tracking RMSE')
        line2 = ax2.plot(speeds, richness_scores, 's-', color='blue', label='Signal Richness')
        
        ax1.set_xlabel('Speed (kph)')
        ax1.set_ylabel('Tracking RMSE (kph)', color='red')
        ax2.set_ylabel('Signal Richness', color='blue')
        ax1.set_title('Data Quality Breakdown')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M_%m%d")
            plot_file = save_path / f"hankel_analysis_plots_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"[Plot] Analysis plots saved to {plot_file}")
        
        plt.show()


def main():
    """Main function to run Hankel matrix analysis."""
    analyzer = HankelMatrixAnalyzer()
    
    try:
        # Load collected PRBS data
        print("Loading PRBS collected data...")
        analyzer.load_collected_data()
        
        # Generate optimal Hankel matrices
        print("Generating optimal Hankel matrices...")
        analyzer.generate_optimal_hankel_matrices()
        
        # Save results
        print("Saving Hankel matrices...")
        saved_file = analyzer.save_hankel_matrices()
        
        # Generate analysis report
        analyzer.generate_analysis_report()
        
        # Generate plots
        print("Generating analysis plots...")
        analyzer.plot_analysis_results()
        
        print(f"\n{'='*80}")
        print("HANKEL MATRIX ANALYSIS COMPLETED!")
        print(f"{'='*80}")
        print(f"Results saved to: {saved_file}")
        print("Next step: Implement speed-scheduled Hankel matrices in DeePC controller")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()