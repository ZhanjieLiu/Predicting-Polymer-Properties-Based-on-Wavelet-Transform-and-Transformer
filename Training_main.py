import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import time

# Initialize lists to store trial hyperparameters and performance results
hyperparameter_results = []

tf.config.set_visible_devices([], 'GPU')

# Read data
data = pd.read_csv('SMILE.csv', encoding='ISO-8859-1')

# SMILES to molecular fingerprint conversion
def smiles_to_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fingerprint)
    except Exception as e:
        print(f"Error in SMILES conversion: {e}")
        return None

# Wavelet transform function
def apply_wavelet_transform(fingerprint, wavelet='haar', level=3):
    coeffs = pywt.wavedec(fingerprint, wavelet, level=level)
    transformed_features = np.concatenate([coeffs[0], coeffs[-1]])
    return transformed_features

# Extract molecular fingerprints and target column
data['fingerprint'] = data['SMILES'].apply(smiles_to_fingerprint)
data = data.dropna(subset=['fingerprint'])

# Target column
target_columns = ["Tg"]
X = np.array(data['fingerprint'].tolist())
y = np.array(data[target_columns])

# Custom R² function for Keras
def r2_keras(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    residual = tf.reduce_sum(tf.square(y_true - y_pred))
    total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - residual / total
    return r2

# HyperModel for Bayesian Optimization
class TransformerHyperModel(kt.HyperModel):
    def build(self, hp):
        # Defining adjustable hyperparameters
        num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
        transformer_units = hp.Int("transformer_units", min_value=64, max_value=256, step=64)
        dense_units = hp.Int("dense_units", min_value=32, max_value=128, step=32)
        dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.3, step=0.1)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        # Adding wavelet and level as hyperparameters
        wavelet = hp.Choice("wavelet", values=['haar', 'db2', 'db3', 'db4', 'sym2', 'coif1'])
        level = hp.Int("level", min_value=1, max_value=4)

        # Input layer
        input_combined = layers.Input(shape=(self.input_shape,))

        # Wavelet transformed features
        wavelet_transformed_features = np.array([apply_wavelet_transform(fp, wavelet=wavelet, level=level) for fp in X])

        # Combine original fingerprints with wavelet transformed features
        X_combined = np.concatenate((X, wavelet_transformed_features), axis=1)

        # Transformer layer
        transformer = layers.Dense(transformer_units, activation='relu')(input_combined)
        transformer = layers.Reshape((1, transformer_units))(transformer)

        transformer_attention = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=transformer_units // num_heads
        )(transformer, transformer)
        transformer = layers.GlobalAveragePooling1D()(transformer_attention)

        # Fully connected layer
        combined = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(0.001))(transformer)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(dropout_rate)(combined)
        output = layers.Dense(1, activation='linear')(combined)

        # Build model
        model = Model(inputs=input_combined, outputs=output)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=[r2_keras])
        
        return model

    def fit(self, hp, model, *args, **kwargs):
        # Before training, store the hyperparameters for the trial
        trial_hyperparameters = {
            'num_heads': hp.get('num_heads'),
            'transformer_units': hp.get('transformer_units'),
            'dense_units': hp.get('dense_units'),
            'dropout_rate': hp.get('dropout_rate'),
            'learning_rate': hp.get('learning_rate'),
            'wavelet': hp.get('wavelet'),
            'level': hp.get('level')
        }
        
        history = model.fit(*args, **kwargs)
        
        # After training, get the performance results (e.g., validation loss)
        val_loss = history.history['val_loss'][-1]
        val_r2 = history.history['val_r2_keras'][-1]
        
        # Evaluate the model on train, validation, and test sets
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate R² and MSE
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # Add the trial's results (hyperparameters + metrics) to the results list
        hyperparameter_results.append({
            **trial_hyperparameters,
            'val_loss': val_loss,
            'val_r2': val_r2,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'train_mse': train_mse,
            'test_r2': test_r2,
            'test_mse': test_mse
        })
        
        return history


# Split data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

# Initialize HyperModel
hypermodel = TransformerHyperModel()
hypermodel.input_shape = X_train.shape[1]

# Bayesian Optimization tuner
tuner = kt.BayesianOptimization(
    hypermodel,
    objective='val_loss',
    max_trials=50,  
    directory='bayesian_tuning',
    project_name='wavelet_transformer_model'
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)

# Run Bayesian Optimization
tuner.search(
    X_train, y_train,
    epochs=200,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Save the hyperparameters and corresponding results to a DataFrame
df_results = pd.DataFrame(hyperparameter_results)

# Save results to CSV
df_results.to_csv('bayesian_optimization_results.csv', index=False)


# Get best model and print best hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]

# Fix: Get best hyperparameters using tuner.oracle.get_best_trials
best_hyperparameters = tuner.oracle.get_best_trials()[0].hyperparameters.values
print("Best Hyperparameters: ", best_hyperparameters)

# Evaluate the model
y_pred_train = best_model.predict(X_train)
y_pred_val = best_model.predict(X_val)
y_pred_test = best_model.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
val_mse = mean_squared_error(y_val, y_pred_val)
test_mse = mean_squared_error(y_test, y_pred_test)

train_r2 = r2_score(y_train, y_pred_train)
val_r2 = r2_score(y_val, y_pred_val)
test_r2 = r2_score(y_test, y_pred_test)

# Record results
results = [{
    'train_r2': train_r2,
    'val_r2': val_r2,
    'test_r2': test_r2,
    'train_mse': train_mse,
    'val_mse': val_mse,
    'test_mse': test_mse
}]
print(results)
