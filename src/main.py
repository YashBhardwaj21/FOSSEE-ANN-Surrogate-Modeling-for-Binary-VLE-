import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import interpolate
import joblib
import tensorflow as tf
import os

class VLEPredictor:
    """VLE prediction system using the trained ANN model"""
    
    def __init__(self):
        self.model = None
        self.scaler_X = None
        
    def load_model_and_scalers(self):
        """Load the trained model and scalers"""
        try:
            # Load the scaler (only X is scaled now)
            self.scaler_X = joblib.load('scaler_X.pkl')
            
            # Load the model
            self.model = tf.keras.models.load_model('models/ethanol_water_vle_model_final.keras')
            
            print("✅ Model and scaler loaded successfully!")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Please make sure you've:")
            print("1. Run the data generation script")
            print("2. Trained the model by running ann.py")
            return False
    
    def prepare_input_features(self, x1, T_K, P_kPa):
        """
        Prepare input features with enhanced engineering (matches training)
        """
        # Original interaction terms
        x1_T = x1 * T_K
        x1_P = x1 * P_kPa
        T_P = T_K * P_kPa
        
        # NEW: Azeotrope-aware features
        distance_to_azeotrope = np.abs(x1 - 0.894)
        azeotrope_interaction = x1 * (1 - x1) * distance_to_azeotrope
        logit_x1 = np.log(x1 / (1 - x1 + 1e-10))  # Avoid division by zero
        
        return np.array([[x1, T_K, P_kPa, x1_T, x1_P, T_P, 
                         distance_to_azeotrope, azeotrope_interaction, logit_x1]])
    
    def predict_vle(self, x1, T_C, P_atm):
        """
        Predict vapor composition for given conditions
        :param x1: Liquid mole fraction of ethanol (0-1)
        :param T_C: Temperature in °C
        :param P_atm: Pressure in atm
        :return: Predicted vapor composition y1
        """
        if self.model is None or self.scaler_X is None:
            print("Model or scaler not loaded. Call load_model_and_scalers() first.")
            return None
        
        # Convert to model input format
        T_K = T_C + 273.15
        P_kPa = P_atm * 101.325
        
        # Create input array with enhanced features
        input_data = self.prepare_input_features(x1, T_K, P_kPa)
        
        # Scale the input
        input_scaled = self.scaler_X.transform(input_data)
        
        # Make prediction (no need to inverse transform y since it's not scaled)
        y_pred = self.model.predict(input_scaled, verbose=0).flatten()[0]
        
        return y_pred
    
    def test_predefined_cases(self):
        """Test the model with predefined test cases"""
        test_cases = [
            # [x1, T_C, P_atm, description]
            [0.01, 95.0, 1.0, "Very low ethanol"],
            [0.05, 90.0, 1.0, "Low ethanol"],
            [0.10, 95.0, 1.0, "Medium-low ethanol"],
            [0.30, 85.0, 1.0, "Medium concentration"],
            [0.50, 80.0, 1.0, "Equal mixture"],
            [0.70, 78.0, 1.0, "High ethanol"],
            [0.85, 78.5, 1.0, "Near azeotrope"],
            [0.894, 78.2, 1.0, "Azeotropic point"],
            [0.95, 78.0, 1.0, "Very high ethanol"],
            [0.30, 75.0, 0.5, "Low pressure"],
            [0.30, 90.0, 2.0, "High pressure"],
        ]
        
        print("\n" + "="*80)
        print("PREDEFINED TEST CASES")
        print("="*80)
        print(f"{'Description':<20} {'x1':<6} {'T(°C)':<6} {'P(atm)':<7} {'Pred y1':<8} {'α':<8}")
        print("-" * 80)
        
        results = []
        for x1, T_C, P_atm, desc in test_cases:
            y1_pred = self.predict_vle(x1, T_C, P_atm)
            
            # Calculate relative volatility if meaningful
            if 0 < x1 < 1 and 0 < y1_pred < 1:
                alpha = (y1_pred / (1 - y1_pred)) / (x1 / (1 - x1))
                alpha_str = f"{alpha:.3f}"
            else:
                alpha_str = "N/A"
            
            results.append([desc, x1, T_C, P_atm, y1_pred, alpha])
            print(f"{desc:<20} {x1:<6.3f} {T_C:<6.1f} {P_atm:<7.2f} {y1_pred:<8.4f} {alpha_str:<8}")
        
        return results
    
    def interactive_prediction(self):
        """Interactive mode for user input"""
        print("\n" + "="*50)
        print("INTERACTIVE VLE PREDICTION")
        print("="*50)
        
        while True:
            try:
                print("\nEnter the following values (or 'quit' to exit):")
                x1 = input("x1 (liquid mole fraction of ethanol, 0-1): ")
                if x1.lower() == 'quit':
                    break
                
                T_C = input("T_C (temperature in °C): ")
                if T_C.lower() == 'quit':
                    break
                
                P_atm = input("P_atm (pressure in atm): ")
                if P_atm.lower() == 'quit':
                    break
                
                # Convert to numbers
                x1 = float(x1)
                T_C = float(T_C)
                P_atm = float(P_atm)
                
                # Validate inputs
                if not (0 <= x1 <= 1):
                    print("❌ x1 must be between 0 and 1")
                    continue
                if T_C < 0 or T_C > 100:
                    print("⚠️  Warning: Temperature should be between 0°C and 100°C")
                if P_atm < 0.1 or P_atm > 10:
                    print("⚠️  Warning: Pressure should be between 0.1 atm and 10 atm")
                
                # Make prediction
                y1_pred = self.predict_vle(x1, T_C, P_atm)
                
                print(f"\nResults:")
                print(f"  Liquid composition (x1): {x1:.4f}")
                print(f"  Temperature: {T_C:.1f}°C")
                print(f"  Pressure: {P_atm:.3f} atm")
                print(f"  Predicted vapor composition (y1): {y1_pred:.4f}")
                print(f"  Vapor contains {y1_pred*100:.1f}% ethanol")
                
                # Calculate relative volatility if meaningful
                if 0 < x1 < 1 and 0 < y1_pred < 1:
                    alpha = (y1_pred / (1 - y1_pred)) / (x1 / (1 - x1))
                    print(f"  Relative volatility (α): {alpha:.3f}")
                
            except ValueError:
                print("❌ Please enter valid numerical values")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def create_validation_plots(self):
        """Create validation plots using test data"""
        try:
            # Load the dataset
            data = pd.read_csv('./dataset/ethanol_water_vle_dataset.csv')
            
            # Prepare features the same way as training (ENHANCED)
            data['x1_T'] = data['x1'] * data['T_K']
            data['x1_P'] = data['x1'] * data['P_kPa']
            data['T_P'] = data['T_K'] * data['P_kPa']
            data['distance_to_azeotrope'] = np.abs(data['x1'] - 0.894)
            data['azeotrope_interaction'] = data['x1'] * (1 - data['x1']) * data['distance_to_azeotrope']
            data['logit_x1'] = np.log(data['x1'] / (1 - data['x1'] + 1e-10))
            
            features = ['x1', 'T_K', 'P_kPa', 'x1_T', 'x1_P', 'T_P', 
                       'distance_to_azeotrope', 'azeotrope_interaction', 'logit_x1']
            X = data[features].values
            y_true = data['y1'].values
            
            # Make predictions for all data points
            X_scaled = self.scaler_X.transform(X)
            y_pred = self.model.predict(X_scaled, verbose=0).flatten()
            
            # Create parity plot
            self._create_parity_plot(y_true, y_pred)
            
            # Create error distribution plot
            self._create_error_plot(y_true, y_pred)
            
            # Create composition profile plot
            self._create_composition_plot(data, y_pred)
            
        except FileNotFoundError:
            print("Dataset file not found. Skipping validation plots.")
    
    def _create_parity_plot(self, y_true, y_pred):
        """Create parity plot"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, s=10)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
        plt.xlabel('True y1')
        plt.ylabel('Predicted y1')
        plt.title('Parity Plot: True vs Predicted Vapor Composition')
        plt.grid(True, alpha=0.3)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Add metrics text
        textstr = f'MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                      verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('parity_plot.png', dpi=300)
        plt.show()
        print(f"✅ Parity plot saved as 'parity_plot.png'")
    
    def _create_error_plot(self, y_true, y_pred):
        """Create error distribution plot"""
        errors = y_pred - y_true
        
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (y_pred - y_true)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        textstr = f'Mean Error: {mean_error:.4f}\nStd Dev: {std_error:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                      verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('error_distribution.png', dpi=300)
        plt.show()
        print(f"✅ Error distribution plot saved as 'error_distribution.png'")
    
    def _create_composition_plot(self, data, y_pred):
        """Create composition profile plot"""
        plt.figure(figsize=(10, 6))
        
        # Sort by x1 for better visualization
        sorted_indices = np.argsort(data['x1'])
        x1_sorted = data['x1'].iloc[sorted_indices]
        y_true_sorted = data['y1'].iloc[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]
        
        plt.plot(x1_sorted, y_true_sorted, 'bo', markersize=4, alpha=0.6, label='True')
        plt.plot(x1_sorted, y_pred_sorted, 'r-', linewidth=2, label='Predicted')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='x=y line')
        
        # Mark azeotropic point
        plt.axvline(x=0.894, color='g', linestyle=':', alpha=0.7, label='Azeotrope (x=0.894)')
        
        plt.xlabel('Liquid Composition (x1)')
        plt.ylabel('Vapor Composition (y1)')
        plt.title('Ethanol-Water VLE Composition Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('composition_profile.png', dpi=300)
        plt.show()
        print(f"✅ Composition profile plot saved as 'composition_profile.png'")

def main():
    """Main function"""
    predictor = VLEPredictor()
    
    # Load model and scalers
    if not predictor.load_model_and_scalers():
        return
    
    print("\nEthanol-Water VLE Prediction System")
    print("===================================")
    
    while True:
        print("\nOptions:")
        print("1. Run predefined test cases")
        print("2. Interactive prediction")
        print("3. Create validation plots")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            predictor.test_predefined_cases()
        
        elif choice == '2':
            predictor.interactive_prediction()
        
        elif choice == '3':
            predictor.create_validation_plots()
        
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()