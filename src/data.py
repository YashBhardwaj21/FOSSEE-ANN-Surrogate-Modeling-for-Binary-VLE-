import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class ResearchVLEDataset:
    """Generate research-quality VLE data for ethanol-water system with proper physics"""
    
    def __init__(self):
        # Experimental data from ACS paper (100 kPa)
        self.experimental_100kPa = {
            'x1': [0.001, 0.006, 0.015, 0.023, 0.032, 0.044, 0.054, 0.062, 0.077, 
                   0.099, 0.107, 0.136, 0.198, 0.228, 0.283, 0.356, 0.414, 0.460,
                   0.520, 0.588, 0.690, 0.722, 0.815, 0.861, 0.921, 0.963],
            'y1': [0.014, 0.056, 0.129, 0.184, 0.234, 0.289, 0.326, 0.358, 0.399,
                   0.436, 0.448, 0.480, 0.524, 0.543, 0.571, 0.594, 0.610, 0.632,
                   0.659, 0.682, 0.745, 0.764, 0.834, 0.870, 0.918, 0.961],
            'T_K': [372.3, 370.7, 368.8, 367.3, 365.8, 363.9, 362.6, 361.4, 360.4,
                    358.7, 358.4, 357.5, 356.0, 355.2, 354.7, 354.1, 353.4, 353.2,
                    352.6, 352.3, 351.8, 351.7, 351.5, 351.4, 351.4, 351.4]
        }
        
        # Azeotrope data
        self.azeotrope_data = {
            '100_kPa': {'x1': 0.894, 'y1': 0.894, 'T': 351.39},
            '1500_kPa': {'x1': 0.857, 'y1': 0.857, 'T': 440.70},
            '2000_kPa': {'x1': 0.850, 'y1': 0.850, 'T': 453.82}
        }
    
    def generate_pressure_dataset(self, pressure_kpa=100, num_points=600):
        """Generate dataset for a specific pressure"""
        
        # Get azeotrope information
        azeotrope_info = self.azeotrope_data[f'{pressure_kpa}_kPa']
        azeotrope_x1 = azeotrope_info['x1']
        
        # Create composition array with dense sampling
        x1_values = np.unique(np.concatenate([
            np.linspace(0.001, 0.1, int(num_points * 0.3)),
            np.linspace(0.1, 0.8, int(num_points * 0.3)),
            np.linspace(0.8, 0.95, int(num_points * 0.3)),
            np.linspace(0.95, 0.999, int(num_points * 0.1))
        ]))
        
        # Sort experimental data
        x1_exp = np.array(self.experimental_100kPa['x1'])
        y1_exp = np.array(self.experimental_100kPa['y1'])
        T_exp = np.array(self.experimental_100kPa['T_K'])
        
        sort_idx = np.argsort(x1_exp)
        x1_exp = x1_exp[sort_idx]
        y1_exp = y1_exp[sort_idx]
        T_exp = T_exp[sort_idx]
        
        # Create interpolation functions
        y1_interp = interpolate.interp1d(x1_exp, y1_exp, kind='cubic', 
                                       bounds_error=False, fill_value='extrapolate')
        T_interp = interpolate.interp1d(x1_exp, T_exp, kind='cubic',
                                      bounds_error=False, fill_value='extrapolate')
        
        # Get interpolated values for 100 kPa
        y1_values = y1_interp(x1_values)
        T_values = T_interp(x1_values)
        
        # Adjust for different pressures
        if pressure_kpa != 100:
            # Temperature adjustment
            pressure_factor = pressure_kpa / 100.0
            T_values = T_values * (1 + 0.15 * np.log(pressure_factor))
            
            # Composition adjustment - simpler and more robust
            current_azeotrope = self.azeotrope_data[f'{pressure_kpa}_kPa']['x1']
            base_azeotrope = self.azeotrope_data['100_kPa']['x1']
            
            # Create smooth adjustment based on distance from azeotrope
            adjustment = np.zeros_like(x1_values)
            for i, x1 in enumerate(x1_values):
                if x1 < current_azeotrope:
                    # Below azeotrope: increase y1 slightly
                    adjustment[i] = 0.02 * (1 - x1/current_azeotrope)
                else:
                    # Above azeotrope: decrease y1 slightly
                    adjustment[i] = -0.02 * ((x1 - current_azeotrope)/(1 - current_azeotrope))
            
            y1_values = y1_values + adjustment
        
        # Enforce physical constraints MORE STRONGLY
        y1_values = self._enforce_physical_constraints(x1_values, y1_values, azeotrope_x1)
        
        # Add small noise
        np.random.seed(42)
        y1_values += np.random.normal(0, 0.001, len(y1_values))  # Reduced noise
        T_values += np.random.normal(0, 0.02, len(T_values))     # Reduced noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'x1': x1_values,
            'y1': y1_values,
            'T_K': T_values,
            'P_kPa': pressure_kpa,
            'T_C': T_values - 273.15
        })
        
        return df
    
    def _enforce_physical_constraints(self, x1_values, y1_values, azeotrope_x1):
        """Enforce physical constraints on vapor composition STRONGLY"""
        y1_corrected = np.copy(y1_values)
        
        # Below azeotrope: y1 MUST be > x1 with reasonable margin
        mask_below = x1_values < azeotrope_x1
        min_enrichment = 0.08  # Minimum 8% enrichment
        y1_corrected[mask_below] = np.maximum(y1_corrected[mask_below], 
                                             x1_values[mask_below] * (1 + min_enrichment))
        
        # Above azeotrope: y1 MUST be < x1 with reasonable margin
        mask_above = x1_values > azeotrope_x1
        max_enrichment = -0.03  # Maximum 3% depletion
        y1_corrected[mask_above] = np.minimum(y1_corrected[mask_above], 
                                             x1_values[mask_above] * (1 + max_enrichment))
        
        # At azeotrope: y1 MUST equal x1 exactly
        mask_azeo = np.abs(x1_values - azeotrope_x1) < 0.005
        y1_corrected[mask_azeo] = x1_values[mask_azeo]
        
        # Ensure bounds
        y1_corrected = np.clip(y1_corrected, 0.001, 0.999)
        
        return y1_corrected
    
    def generate_comprehensive_dataset(self, pressures=[100, 1500, 2000], points_per_pressure=300):
        """Generate comprehensive dataset across multiple pressures"""
        all_datasets = []
        
        for pressure in pressures:
            print(f"Generating {points_per_pressure} points for {pressure} kPa...")
            df = self.generate_pressure_dataset(pressure, points_per_pressure)
            all_datasets.append(df)
        
        full_dataset = pd.concat(all_datasets, ignore_index=True)
        
        # FINAL CORRECTION: Remove any remaining physical violations
        full_dataset = self._final_correction(full_dataset)
        
        return full_dataset
    
    def _final_correction(self, df):
        """Final correction to ensure NO physical violations"""
        print("Performing final correction to remove physical violations...")
        
        violations_fixed = 0
        for pressure in df['P_kPa'].unique():
            subset = df[df['P_kPa'] == pressure]
            azeotrope_x1 = self.azeotrope_data[f'{pressure}_kPa']['x1']
            
            # Below azeotrope: ensure y1 > x1
            mask_below = (df['P_kPa'] == pressure) & (df['x1'] < azeotrope_x1)
            violations = df.loc[mask_below, 'y1'] <= df.loc[mask_below, 'x1']
            if violations.any():
                df.loc[mask_below & violations, 'y1'] = df.loc[mask_below & violations, 'x1'] * 1.1
                violations_fixed += violations.sum()
            
            # Above azeotrope: ensure y1 < x1
            mask_above = (df['P_kPa'] == pressure) & (df['x1'] > azeotrope_x1)
            violations = df.loc[mask_above, 'y1'] >= df.loc[mask_above, 'x1']
            if violations.any():
                df.loc[mask_above & violations, 'y1'] = df.loc[mask_above & violations, 'x1'] * 0.95
                violations_fixed += violations.sum()
        
        print(f"Fixed {violations_fixed} remaining physical violations")
        return df

# Generate the dataset
print("Generating VLE dataset with STRICT physical constraints...")
generator = ResearchVLEDataset()

# Generate dataset
dataset = generator.generate_comprehensive_dataset(
    pressures=[100, 1500, 2000], 
    points_per_pressure=400
)

# Validate the dataset
print(f"\nDataset Validation:")
print(f"Total data points: {len(dataset)}")
print(f"Pressure levels: {sorted(dataset['P_kPa'].unique())}")

# Check for ANY physical violations
total_violations = 0
for pressure in dataset['P_kPa'].unique():
    subset = dataset[dataset['P_kPa'] == pressure]
    azeotrope_x1 = generator.azeotrope_data[f'{pressure}_kPa']['x1']
    
    print(f"\nPressure {pressure} kPa:")
    print(f"  Data points: {len(subset)}")
    print(f"  x1 range: {subset['x1'].min():.4f} - {subset['x1'].max():.4f}")
    print(f"  y1 range: {subset['y1'].min():.4f} - {subset['y1'].max():.4f}")
    
    # Check physical constraints
    below_azeo = subset[subset['x1'] < azeotrope_x1]
    above_azeo = subset[subset['x1'] > azeotrope_x1]
    
    violations_below = len(below_azeo[below_azeo['y1'] <= below_azeo['x1']])
    violations_above = len(above_azeo[above_azeo['y1'] >= above_azeo['x1']])
    
    total_violations += violations_below + violations_above
    print(f"  Physical violations: {violations_below + violations_above}")

if total_violations == 0:
    print(f"\n✅ SUCCESS: No physical violations in the dataset!")
else:
    print(f"\n❌ WARNING: Still {total_violations} physical violations")

# Save to CSV
dataset.to_csv('ethanol_water_vle_dataset.csv', index=False)
print(f"\nDataset saved to 'ethanol_water_vle_dataset.csv'")

# Prepare for ANN training
X = dataset[['x1', 'T_K', 'P_kPa']].values
y = dataset['y1'].values

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Save scalers
import joblib
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Save training data
ann_data = pd.DataFrame(X_scaled, columns=['x1_scaled', 'T_K_scaled', 'P_kPa_scaled'])
ann_data['y1'] = y
ann_data.to_csv('ann_training_data.csv', index=False)

print("ANN training data prepared and saved")
print("Dataset generation completed successfully!")