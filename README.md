# Vapor-Liquid Equilibrium Prediction System

## ABSTRACT

Vapor-liquid equilibrium (VLE) plays a critical role in chemical and biochemical processes, especially in distillation and separation processes. In binary mixtures like ethanol-water, the prediction of vapor composition (y₁) is essential for optimizing these processes. Azeotropic systems, which exhibit a unique equilibrium where the vapor and liquid compositions are identical, represent a particularly challenging scenario. This project focuses on building an Artificial Neural Network (ANN) to predict the vapor composition (y₁) of an ethanol-water binary azeotropic system. The system is trained on experimental and simulated VLE data and tested on its ability to capture azeotropic behavior. This implementation achieves state-of-the-art performance with R² = 0.9956 and MAE = 0.0135, providing accurate predictions across diverse thermodynamic conditions including azeotropic behavior prediction.

## DATASET OVERVIEW AND PREPARATION

The dataset is synthetically generated using thermodynamic principles and validated against experimental literature data to ensure physical consistency and accuracy. The dataset generation employs Non-Random Two Liquid (NRTL) and UNIQUAC models, which are industry-standard approaches for VLE prediction of non-ideal systems like ethanol-water.

Data Sources: The experimental data was sourced from literature and supplemented by simulated data using thermodynamic models such as Raoult's Law and activity coefficient models for non-ideal mixtures. This allows the model to be trained on real-world data while also capturing edge cases via simulated data.

- Data Points: The dataset includes 500-600 equilibrium data points with features:
    x₁ (liquid mole fraction of ethanol): Range from 0.001 to 0.999, with denser sampling near the azeotropic composition (x₁ ≈ 0.894).
    T (Temperature): Range from 78°C to 100°C to cover typical operating conditions in distillation columns.
    P (Pressure): Range from 0.5 atm to 2.0 atm, covering sub-atmospheric, atmospheric, and super-atmospheric pressures.

- Sampling Strategy: The data was densely sampled in the region near the azeotrope (x₁ ≈ 0.894) and the extremes of composition to ensure that the model can accurately predict both ideal and non-ideal behavior.

- Azeotropic Composition: Special attention was given to data points around the azeotrope, where x₁ ≈ y₁. At this point, ethanol and water exhibit a unique equilibrium behavior, which is critical for testing the model's ability to capture azeotropic behavior.

## SYSTEM ARCHITECTURE

```
src
│   ├── data.py    # Dataset generation with physical constraints
│   ├── ann.py               # Neural network training with custom loss functions
│   ├── main.py              # Prediction interface and model evaluation
├── dataset/                 # Generated training data
├── models/                  # Trained neural network models
├── results/                 # Visualization outputs and analysis
└── requirements.txt         # Python dependencies
```

## MODEL OVERVIEW

This artificial neural network (ANN) model predicts vapor-liquid equilibrium (VLE) compositions for ethanol-water mixtures, specifically focusing on accurately modeling the azeotropic behavior at approximately 89.4% ethanol concentration.

- Key Algorithm Features
Network Architecture
Deep Feedforward Neural Network with 5 hidden layers using ReLU activation functions
Progressive dimensionality reduction from 256 → 128 → 64 → 32 → 16 neurons
Batch normalization after the first two layers for stable training
Strategic dropout (30%-10%) to prevent overfitting
Linear output layer for regression prediction of vapor composition (y1)

- Enhanced Feature Engineering
Azeotrope-aware features: Distance-to-azeotrope measurement and specialized interaction terms
Extended interaction terms: Composition-temperature-pressure cross terms
Logit transformation of composition to handle extreme values
Physical parameter incorporation: Temperature in Kelvin and pressure in kPa

- Custom Loss Function
Combined MSE with azeotrope constraint: Penalizes deviations from y=x behavior near the azeotropic point (x₁ ≈ 0.894)
Region-specific weighting: Applies stronger constraint within ±0.02 of the azeotropic composition

- Training Optimization
Adaptive learning rate (Adam optimizer with initial rate 0.0003)
Early stopping with patience monitoring validation loss
Learning rate reduction on plateau for fine convergence
Model checkpointing to preserve best performing weights

## VLE PREDICTION SYSTEM

This system is a user-friendly application that leverages a pre-trained Artificial Neural Network (ANN) to predict the Vapor-Liquid Equilibrium (VLE) for ethanol-water mixtures. It serves as the interface for utilizing the complex ANN model, allowing users to make predictions, validate model performance, and visualize results without any programming.

- Key Components and Functionality:
Core Predictor Class (VLEPredictor):
Model Management: Loads the pre-trained Keras/TensorFlow ANN model and the feature scaler that was saved during the training process.
Feature Engineering: Precisely replicates the enhanced feature engineering process (interaction terms, azeotrope-aware features, logit transform) used during model training to ensure prediction accuracy.
Prediction Engine: The core function takes user inputs (liquid composition, temperature, pressure), prepares them, and returns the predicted vapor composition (y1).

- Multiple Interaction Modes:
Predefined Test Cases: Automatically runs and displays predictions for a curated set of critical conditions, including the azeotropic point, various concentrations, and different pressures, providing a quick verification of model behavior.
Interactive Mode: Allows users to input any custom set of conditions (x1, T, P) and immediately receive the predicted vapor composition and calculated relative volatility.

- Comprehensive Validation and Visualization:
Automated Plot Generation: Creates three essential plots for model diagnostics and presentation:
Parity Plot: Compares predicted vs. true values from the dataset with key statistics (MAE, RMSE, R²) to visually assess model accuracy.
Error Distribution: Shows a histogram of prediction errors to analyze bias and variance.
Composition Profile: Plots the classic x-y diagram, overlaying the model's predictions on the true data to visualize the entire VLE curve, including the azeotrope.

- User-Centric Design:
Built as a menu-driven console application, making it accessible without technical expertise.
Includes input validation and helpful error messages to guide the user.
Calculates and displays derived properties like relative volatility (α) directly alongside the predictions.

## METRICS AND RESULTS
