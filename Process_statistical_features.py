#!/usr/bin/env python3
"""
Complete Statistical Feature Extraction for All Capture-24 Subjects
==================================================================

This script processes all 151 subjects (P001.npz to P151.npz) and extracts
the complete set of 56 statistical features for each window.

Author: MiniMax Agent
Date: 2025-11-06
Usage: python process_all_subjects_statistical_features.py

Requirements:
- NumPy, Pandas, SciPy
- Input directory: E:/HAR_datasets_codes/processed_minimal/
- Output: comprehensive_features_table.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')

class Capture24FeatureExtractor:
    """
    Complete statistical feature extractor for Capture-24 accelerometer data.
    Extracts 56 features across 8 categories for each window.
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate: Sampling frequency in Hz (default: 100Hz for Capture-24)
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Feature names for the 56 statistical features
        self.feature_names = [
            # Basic Statistics (9)
            'x_mean', 'x_std', 'x_range',
            'y_mean', 'y_std', 'y_range', 
            'z_mean', 'z_std', 'z_range',
            
            # Cross-Axis Correlations (3)
            'corr_xy', 'corr_xz', 'corr_yz',
            
            # Magnitude Statistics (7)
            'mag_mean', 'mag_std', 'mag_range', 'mag_mad',
            'mag_kurtosis', 'mag_skew', 'mag_median',
            
            # Quantile Features (20)
            'x_min', 'x_max', 'x_median', 'x_q25', 'x_q75',
            'y_min', 'y_max', 'y_median', 'y_q25', 'y_q75',
            'z_min', 'z_max', 'z_median', 'z_q25', 'z_q75',
            'mag_min', 'mag_max', 'mag_median', 'mag_q25', 'mag_q75',
            
            # Temporal Features (1)
            'autocorr_lag1',
            
            # Spectral Features (5)
            'dominant_freq_1', 'dominant_freq_2', 'dominant_freq_power_1',
            'dominant_freq_power_2', 'spectral_entropy',
            
            # Peak Features (2)
            'peak_count', 'peak_prominence',
            
            # Angular Features (9)
            'gravity_roll_std', 'gravity_pitch_std', 'gravity_yaw_std',
            'dynamic_roll_std', 'dynamic_pitch_std', 'dynamic_yaw_std',
            'gravity_roll_mean', 'gravity_pitch_mean', 'gravity_yaw_mean'
        ]
        
        print(f"Initialized feature extractor with {len(self.feature_names)} features")
    
    def load_subject_data(self, file_path: str) -> Optional[Dict]:
        """
        Load subject data from NPZ file with comprehensive error handling.
        
        Args:
            file_path: Path to the subject's NPZ file
            
        Returns:
            Dictionary with subject data or None if loading failed
        """
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Expected arrays in the NPZ file
            expected_arrays = ['windows', 'labels_str', 'times_epoch_ns', 
                             'first_ts_epoch_ns', 'pid', 'window_size', 'signal_rate']
            
            # Check if all expected arrays are present
            missing_arrays = [arr for arr in expected_arrays if arr not in data]
            if missing_arrays:
                print(f"Warning: Missing arrays {missing_arrays} in {file_path}")
                return None
            
            subject_data = {
                'windows': data['windows'],  # Shape: (n_windows, 1000, 3)
                'labels_str': data['labels_str'],
                'pid': data['pid'] if data['pid'].ndim > 0 else data['pid'].item(),
                'window_size': data['window_size'] if data['window_size'].ndim > 0 else data['window_size'].item(),
                'signal_rate': data['signal_rate'] if data['signal_rate'].ndim > 0 else data['signal_rate'].item(),
                'times_epoch_ns': data['times_epoch_ns'] if 'times_epoch_ns' in data else None
            }
            
            print(f"Loaded subject {subject_data['pid']}: {subject_data['windows'].shape[0]} windows")
            return subject_data
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None
    
    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Z-score normalization with safety checks.
        
        Args:
            signal_data: Input signal array
            
        Returns:
            Normalized signal array
        """
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        
        # Avoid division by zero
        if std_val < 1e-8:
            return np.zeros_like(signal_data)
        
        normalized = (signal_data - mean_val) / std_val
        
        # Clip extreme values (safety measure)
        return np.clip(normalized, -10, 10)
    
    def extract_gravity_component(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract gravity component using 4th order Butterworth low-pass filter at 0.5Hz.
        
        Args:
            signal_data: Input signal (either x, y, or z axis)
            
        Returns:
            Gravity component of the signal
        """
        try:
            # Design 4th order Butterworth low-pass filter at 0.5Hz
            nyquist = self.sampling_rate / 2
            cutoff = 0.5 / nyquist
            b, a = signal.butter(4, cutoff, btype='low')
            
            # Apply zero-phase filtering
            gravity = signal.filtfilt(b, a, signal_data)
            return gravity
            
        except Exception as e:
            print(f"Warning: Could not extract gravity component: {e}")
            return np.zeros_like(signal_data)
    
    def compute_statistical_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract all 56 statistical features from a single window.
        
        Args:
            window: Accelerometer data window (1000, 3) - [x, y, z]
            
        Returns:
            Array of 56 statistical features
        """
        # Validate input
        if window.shape != (1000, 3):
            raise ValueError(f"Expected window shape (1000, 3), got {window.shape}")
        
        # Extract axes
        x, y, z = window[:, 0], window[:, 1], window[:, 2]
        
        # Normalize each axis
        x_norm = self.normalize_signal(x)
        y_norm = self.normalize_signal(y)
        z_norm = self.normalize_signal(z)
        
        # Calculate magnitude
        mag = np.sqrt(x**2 + y**2 + z**2)
        mag_norm = self.normalize_signal(mag)
        
        # Calculate gravity components
        gravity_x = self.extract_gravity_component(x)
        gravity_y = self.extract_gravity_component(y)
        gravity_z = self.extract_gravity_component(z)
        
        # Calculate dynamic components (total - gravity)
        dynamic_x = x - gravity_x
        dynamic_y = y - gravity_y
        dynamic_z = z - gravity_z
        
        features = []
        
        # 1. Basic Statistics (9 features)
        features.extend([
            np.mean(x), np.std(x), np.ptp(x),  # x axis
            np.mean(y), np.std(y), np.ptp(y),  # y axis
            np.mean(z), np.std(z), np.ptp(z)   # z axis
        ])
        
        # 2. Cross-Axis Correlations (3 features)
        try:
            features.extend([
                np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0,
                np.corrcoef(x, z)[0, 1] if len(x) > 1 else 0,
                np.corrcoef(y, z)[0, 1] if len(y) > 1 else 0
            ])
        except:
            features.extend([0, 0, 0])
        
        # 3. Magnitude Statistics (7 features)
        mag_median = np.median(mag)
        features.extend([
            np.mean(mag), np.std(mag), np.ptp(mag), np.median(np.abs(mag - mag_median)),
            stats.kurtosis(mag), stats.skew(mag), mag_median
        ])
        
        # 4. Quantile Features (20 features)
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        for axis_data in [x, y, z, mag]:
            axis_quants = np.quantile(axis_data, quantiles)
            features.extend(axis_quants)
        
        # 5. Temporal Features (1 feature)
        try:
            autocorr = np.corrcoef(mag[:-1], mag[1:])[0, 1]
            features.append(autocorr if not np.isnan(autocorr) else 0)
        except:
            features.append(0)
        
        # 6. Spectral Features (5 features)
        try:
            # Compute FFT of magnitude signal
            fft_mag = np.abs(rfft(mag_norm))
            freqs = rfftfreq(len(mag_norm), 1/self.sampling_rate)
            
            # Find dominant frequencies (excluding DC component)
            fft_mag_no_dc = fft_mag[1:]  # Remove DC component
            freqs_no_dc = freqs[1:]
            
            if len(fft_mag_no_dc) > 0:
                # Get top 2 dominant frequencies and their powers
                top_indices = np.argsort(fft_mag_no_dc)[-2:]
                top_freqs = freqs_no_dc[top_indices]
                top_powers = fft_mag_no_dc[top_indices]
                
                dominant_freq_1 = top_freqs[-1] if len(top_freqs) > 0 else 0
                dominant_freq_2 = top_freqs[-2] if len(top_freqs) > 1 else 0
                dominant_freq_power_1 = top_powers[-1] if len(top_powers) > 0 else 0
                dominant_freq_power_2 = top_powers[-2] if len(top_powers) > 1 else 0
                
                # Spectral entropy
                psd = fft_mag_no_dc**2
                psd_norm = psd / (np.sum(psd) + 1e-8)
                spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
            else:
                dominant_freq_1 = dominant_freq_2 = 0
                dominant_freq_power_1 = dominant_freq_power_2 = 0
                spectral_entropy = 0
            
            features.extend([dominant_freq_1, dominant_freq_2, 
                           dominant_freq_power_1, dominant_freq_power_2, spectral_entropy])
        except Exception as e:
            features.extend([0, 0, 0, 0, 0])
        
        # 7. Peak Features (2 features)
        try:
            # Find peaks in magnitude signal
            peaks, properties = signal.find_peaks(mag, prominence=0.1)
            peak_count = len(peaks)
            
            # Calculate peak prominence statistics
            if len(properties) > 0 and 'prominences' in properties:
                peak_prominence = np.mean(properties['prominences'])
            else:
                peak_prominence = 0
                
            features.extend([peak_count, peak_prominence])
        except:
            features.extend([0, 0])
        
        # 8. Angular Features (9 features)
        try:
            # Gravity angles
            gravity_roll = np.arctan2(gravity_y, gravity_z)
            gravity_pitch = np.arctan2(-gravity_x, np.sqrt(gravity_y**2 + gravity_z**2))
            gravity_yaw = np.arctan2(np.sqrt(gravity_x**2 + gravity_y**2), gravity_z)
            
            # Dynamic angles
            dynamic_roll = np.arctan2(dynamic_y, dynamic_z)
            dynamic_pitch = np.arctan2(-dynamic_x, np.sqrt(dynamic_y**2 + dynamic_z**2))
            dynamic_yaw = np.arctan2(np.sqrt(dynamic_x**2 + dynamic_y**2), dynamic_z)
            
            features.extend([
                np.std(gravity_roll), np.std(gravity_pitch), np.std(gravity_yaw),
                np.std(dynamic_roll), np.std(dynamic_pitch), np.std(dynamic_yaw),
                np.mean(gravity_roll), np.mean(gravity_pitch), np.mean(gravity_yaw)
            ])
        except:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Convert to numpy array and handle any remaining NaN/inf values
        features = np.array(features)
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        
        # Final validation
        if len(features) != 56:
            raise ValueError(f"Expected 56 features, got {len(features)}")
        
        return features
    
    def process_subject(self, subject_file: str) -> Optional[pd.DataFrame]:
        """
        Process a single subject and extract features for all windows.
        
        Args:
            subject_file: Path to subject's NPZ file
            
        Returns:
            DataFrame with features for all windows or None if processing failed
        """
        subject_data = self.load_subject_data(subject_file)
        if subject_data is None:
            return None
        
        windows = subject_data['windows']
        labels = subject_data['labels_str']
        pid = subject_data['pid']
        
        if windows.ndim != 3 or windows.shape[1] != 1000 or windows.shape[2] != 3:
            print(f"Warning: Unexpected window shape {windows.shape} for subject {pid}")
            return None
        
        print(f"Processing {len(windows)} windows for subject {pid}...")
        
        # Extract features for all windows
        all_features = []
        
        for i, window in enumerate(windows):
            try:
                features = self.compute_statistical_features(window)
                all_features.append(features)
                
                # Progress update every 1000 windows
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(windows)} windows")
                    
            except Exception as e:
                print(f"Error processing window {i} for subject {pid}: {e}")
                # Fill with zeros for failed windows
                all_features.append(np.zeros(56))
        
        # Create DataFrame
        feature_df = pd.DataFrame(all_features, columns=self.feature_names)
        
        # Add metadata columns
        feature_df['subject_id'] = pid
        feature_df['window_id'] = range(len(feature_df))
        feature_df['activity_label'] = labels
        feature_df['activity_id'] = pd.Categorical(labels).codes
        
        print(f"Completed subject {pid}: {len(feature_df)} windows processed")
        return feature_df
    
    def process_all_subjects(self, input_dir: str, output_file: str) -> Dict:
        """
        Process all subjects and create comprehensive feature table.
        
        Args:
            input_dir: Directory containing subject NPZ files
            output_file: Output CSV file path
            
        Returns:
            Processing summary dictionary
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        print(f"Starting batch processing of subjects from: {input_dir}")
        print(f"Output will be saved to: {output_file}")
        
        all_dataframes = []
        processing_summary = {
            'total_subjects': 0,
            'processed_subjects': 0,
            'failed_subjects': 0,
            'total_windows': 0,
            'subjects': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        # Process subjects P001 to P151
        for i in range(1, 152):
            subject_id = f"P{i:03d}"
            subject_file = input_path / f"{subject_id}.npz"
            
            processing_summary['total_subjects'] += 1
            
            if not subject_file.exists():
                print(f"File not found: {subject_file}")
                processing_summary['failed_subjects'] += 1
                continue
            
            try:
                # Process subject
                subject_df = self.process_subject(str(subject_file))
                
                if subject_df is not None and not subject_df.empty:
                    all_dataframes.append(subject_df)
                    processing_summary['processed_subjects'] += 1
                    processing_summary['total_windows'] += len(subject_df)
                    processing_summary['subjects'].append({
                        'subject_id': subject_id,
                        'n_windows': len(subject_df),
                        'success': True
                    })
                else:
                    processing_summary['failed_subjects'] += 1
                    processing_summary['subjects'].append({
                        'subject_id': subject_id,
                        'n_windows': 0,
                        'success': False
                    })
                    
            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")
                processing_summary['failed_subjects'] += 1
                processing_summary['subjects'].append({
                    'subject_id': subject_id,
                    'n_windows': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Combine all dataframes
        if all_dataframes:
            print(f"\nCombining data from {len(all_dataframes)} subjects...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Save to CSV
            print(f"Saving combined data to {output_file}...")
            combined_df.to_csv(output_file, index=False)
            
            # Print summary statistics
            print("\n" + "="*60)
            print("BATCH PROCESSING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total subjects found: {processing_summary['total_subjects']}")
            print(f"Successfully processed: {processing_summary['processed_subjects']}")
            print(f"Failed to process: {processing_summary['failed_subjects']}")
            print(f"Total windows extracted: {processing_summary['total_windows']:,}")
            print(f"Output file: {output_file}")
            print(f"Table shape: {combined_df.shape}")
            
            # Activity distribution
            activity_counts = combined_df['activity_label'].value_counts()
            print(f"\nActivity distribution:")
            for activity, count in activity_counts.items():
                percentage = (count / len(combined_df)) * 100
                print(f"  {activity}: {count:,} ({percentage:.1f}%)")
            
            # Memory usage
            memory_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
            print(f"\nMemory usage: {memory_mb:.1f} MB")
            
        else:
            print("No data was successfully processed!")
            combined_df = pd.DataFrame()
        
        # Finalize summary
        processing_summary['processing_time'] = time.time() - start_time
        processing_summary['output_file'] = output_file
        processing_summary['final_shape'] = combined_df.shape if not combined_df.empty else None
        
        # Save processing summary
        summary_file = output_file.replace('.csv', '_processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(processing_summary, f, indent=2)
        
        print(f"\nProcessing summary saved to: {summary_file}")
        print(f"Total processing time: {processing_summary['processing_time']:.2f} seconds")
        
        return processing_summary


def main():
    """
    Main execution function.
    """
    print("Capture-24 Statistical Feature Extraction Tool")
    print("=" * 50)
    
    # Configuration
    INPUT_DIRECTORY = r"E:\HAR_datasets_codes\processed_minimal"
    OUTPUT_FILE = r"comprehensive_features_table.csv"
    
    print(f"Input directory: {INPUT_DIRECTORY}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Expected subjects: P001 to P151")
    
    try:
        # Initialize feature extractor
        extractor = Capture24FeatureExtractor()
        
        # Process all subjects
        summary = extractor.process_all_subjects(INPUT_DIRECTORY, OUTPUT_FILE)
        
        if summary['processed_subjects'] > 0:
            print(f"\n✅ SUCCESS: Extracted features from {summary['processed_subjects']} subjects")
            print(f"📊 Total windows: {summary['total_windows']:,}")
            print(f"💾 Output saved to: {OUTPUT_FILE}")
            print(f"📈 Feature table shape: {summary['final_shape']}")
        else:
            print("\n❌ FAILURE: No subjects were successfully processed")
            
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
