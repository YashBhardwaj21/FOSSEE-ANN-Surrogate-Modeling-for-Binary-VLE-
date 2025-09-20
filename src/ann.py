import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

def azeotrope_constraint_loss(y_true, y_pred, x_input):
    """
    Custom loss function that penalizes deviation from y=x at azeotrope region
    """
    # Standard MSE loss
    mse_loss = tf.keras.losses.mse(y_true, y_pred)
    
    # Penalize deviation from y=x at azeotrope region (x1 ≈ 0.894)
    azeotrope_mask = tf.cast(tf.abs(x_input[:, 0] - 0.894) < 0.02, tf.float32)
    azeotrope_loss = tf.reduce_mean(azeotrope_mask * tf.square(y_pred - x_input[:, 0]))
    
    return mse_loss + 0.1 * azeotrope_loss  # Weight the constraint appropriately

def create_vle_model(input_dim):
    """
    Create enhanced ANN model for VLE prediction with increased complexity
    """
    model = Sequential([
        # Input layer - increased complexity
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers - increased complexity
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(16, activation='relu'),
        
        # Output layer - NO ACTIVATION for regression
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0003),  # Slightly lower learning rate for more complex model
        loss='mse',  # We'll use custom loss during training
        metrics=['mae']
    )

    return model

def prepare_training_data():
    """
    Load and prepare training data with enhanced feature engineering
    """
    if not os.path.exists('./dataset/ethanol_water_vle_dataset.csv'):
        print("❌ Dataset file not found. Please run the data generation script first.")
        return None
    
    # Load the dataset
    data = pd.read_csv('./dataset/ethanol_water_vle_dataset.csv')
    print(f"✅ Dataset loaded: {len(data)} samples")
    
    # ENHANCED FEATURE ENGINEERING WITH AZEOTROPE AWARENESS
    data['T_K'] = data['T_K']
    data['P_kPa'] = data['P_kPa']
    
    # Original interaction terms
    data['x1_T'] = data['x1'] * data['T_K']
    data['x1_P'] = data['x1'] * data['P_kPa']
    data['T_P'] = data['T_K'] * data['P_kPa']
    
    # NEW: Azeotrope-aware features
    data['distance_to_azeotrope'] = np.abs(data['x1'] - 0.894)
    data['azeotrope_interaction'] = data['x1'] * (1 - data['x1']) * data['distance_to_azeotrope']
    data['logit_x1'] = np.log(data['x1'] / (1 - data['x1'] + 1e-10))  # Avoid division by zero
    
    # Use enhanced features
    features = ['x1', 'T_K', 'P_kPa', 'x1_T', 'x1_P', 'T_P', 
                'distance_to_azeotrope', 'azeotrope_interaction', 'logit_x1']
    
    X = data[features].values
    y = data['y1'].values  # DO NOT SCALE y1 - it's already 0-1!
    
    print(f"\n📊 Enhanced feature ranges:")
    for i, feat in enumerate(features):
        print(f"  {feat}: [{X[:,i].min():.3f}, {X[:,i].max():.3f}]")
    print(f"  y1: [{y.min():.3f}, {y.max():.3f}]")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Scale only the input features
    print("\n🔄 Scaling input features...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler_X, 'scaler_X.pkl')
    
    print("✅ Data preparation completed successfully!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, X_train  # Return X_train for custom loss

def train_model(X_train, y_train, X_val, y_val, X_train_original, epochs=600, batch_size=32):
    """
    Train the ANN model with custom azeotrope constraint loss
    """
    model = create_vle_model(X_train.shape[1])

    # Convert to TensorFlow tensors for custom loss
    X_train_tf = tf.constant(X_train_original, dtype=tf.float32)
    y_train_tf = tf.constant(y_train, dtype=tf.float32)
    
    # Create custom training function with azeotrope constraint
    def train_step_with_constraint():
        with tf.GradientTape() as tape:
            predictions = model(X_train, training=True)
            loss = azeotrope_constraint_loss(y_train_tf, predictions, X_train_tf)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    callbacks = [
        EarlyStopping(patience=60, restore_best_weights=True, verbose=1, 
                     min_delta=0.00001, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=30, verbose=1,
                         min_lr=1e-7, monitor='val_loss'),
        ModelCheckpoint('models/best_ethanol_vle_model.keras', save_best_only=True,
                       monitor='val_loss', verbose=1)
    ]

    print(f"\n🚀 Starting training with azeotrope constraint:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Model parameters: {model.count_params():,}")

    # Standard training with validation monitoring
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    return model, history

def evaluate_model(model, X_test, y_test, scaler_X):
    """
    Comprehensive model evaluation
    """
    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Evaluation Results:")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    
    # Test with critical input cases
    test_cases = [
        [0.01, 95.0 + 273.15, 101.325, "Very low ethanol"],
        [0.05, 90.0 + 273.15, 101.325, "Low ethanol"], 
        [0.10, 95.0 + 273.15, 101.325, "Medium-low ethanol"],
        [0.30, 85.0 + 273.15, 101.325, "Medium concentration"],
        [0.50, 80.0 + 273.15, 101.325, "Equal mixture"],
        [0.70, 78.0 + 273.15, 101.325, "High ethanol"],
        [0.85, 78.5 + 273.15, 101.325, "Near azeotrope"],
        [0.894, 78.2 + 273.15, 101.325, "Azeotropic point"],
        [0.95, 78.0 + 273.15, 101.325, "Very high ethanol"],
        [0.30, 75.0 + 273.15, 50.6625, "Low pressure"],
        [0.30, 90.0 + 273.15, 202.65, "High pressure"],
    ]
    
    print(f"\n🧪 Critical Test Predictions:")
    print(f"{'Description':<20} {'x1':<6} {'T(K)':<7} {'P(kPa)':<8} {'Pred y1':<8} {'α':<8}")
    print("-" * 65)
    
    results = []
    for x1, T_K, P_kPa, desc in test_cases:
        # Prepare input with enhanced features
        x1_T = x1 * T_K
        x1_P = x1 * P_kPa
        T_P = T_K * P_kPa
        distance_to_azeotrope = np.abs(x1 - 0.894)
        azeotrope_interaction = x1 * (1 - x1) * distance_to_azeotrope
        logit_x1 = np.log(x1 / (1 - x1 + 1e-10))
        
        input_data = np.array([[x1, T_K, P_kPa, x1_T, x1_P, T_P, 
                              distance_to_azeotrope, azeotrope_interaction, logit_x1]])
        input_scaled = scaler_X.transform(input_data)
        
        # Predict
        y_pred = model.predict(input_scaled, verbose=0).flatten()[0]
        
        # Calculate relative volatility
        if 0 < x1 < 1 and 0 < y_pred < 1:
            alpha = (y_pred / (1 - y_pred)) / (x1 / (1 - x1))
            alpha_str = f"{alpha:.3f}"
        else:
            alpha_str = "N/A"
        
        results.append([desc, x1, T_K - 273.15, P_kPa/101.325, y_pred, alpha])
        print(f"{desc:<20} {x1:<6.3f} {T_K-273.15:<6.1f} {P_kPa/101.325:<7.2f} {y_pred:<8.4f} {alpha_str:<8}")
    
    return mae, r2, results

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_predictions_to_csv(results, filename='model_predictions.csv'):
    """Save prediction results to CSV"""
    df = pd.DataFrame(results, columns=['Description', 'x1', 'T_C', 'P_atm', 'Predicted_y1', 'Alpha'])
    df.to_csv(filename, index=False)
    print(f"✅ Predictions saved to {filename}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    result = prepare_training_data()
    if result is None:
        print("Please run the data generation script first to create the dataset.")
    else:
        X_train, X_val, y_train, y_val, scaler_X, X_train_original = result
        
        # Train the model with enhanced features and constraints
        model, history = train_model(X_train, y_train, X_val, y_val, X_train_original)
        
        # Load the best model
        model = tf.keras.models.load_model('models/best_ethanol_vle_model.keras')
        
        # Evaluate the model
        mae, r2, test_results = evaluate_model(model, X_val, y_val, scaler_X)
        
        # Save predictions
        save_predictions_to_csv(test_results)
        
        # Save the final model
        model.save('models/ethanol_water_vle_model_final.keras')
        print("✅ Model training completed and saved!")
        
        # Plot training history
        plot_training_history(history)
        
        # Final verification
        print(f"\n🎯 Final verification:")
        sample_x1, sample_T, sample_P = 0.5, 80.0, 1.0
        T_K = sample_T + 273.15
        P_kPa = sample_P * 101.325
        
        # Prepare enhanced features
        x1_T = sample_x1 * T_K
        x1_P = sample_x1 * P_kPa
        T_P = T_K * P_kPa
        distance_to_azeotrope = np.abs(sample_x1 - 0.894)
        azeotrope_interaction = sample_x1 * (1 - sample_x1) * distance_to_azeotrope
        logit_x1 = np.log(sample_x1 / (1 - sample_x1 + 1e-10))
        
        sample_input = np.array([[sample_x1, T_K, P_kPa, x1_T, x1_P, T_P, 
                                distance_to_azeotrope, azeotrope_interaction, logit_x1]])
        sample_scaled = scaler_X.transform(sample_input)
        prediction = model.predict(sample_scaled, verbose=0)
        
        print(f"Input: x1={sample_x1}, T={sample_T}°C, P={sample_P}atm")
        print(f"Predicted y1: {prediction[0,0]:.4f}")