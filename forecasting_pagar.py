import pandas as pd
import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from concurrent.futures import ThreadPoolExecutor
from tensorflow.data import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU if necessary

def get_date_range_from_directory(directory):
    filenames = os.listdir(directory)
    filenames = [f for f in filenames if f.endswith('.parquet')]

    dates = []
    for filename in filenames:
        try:
            date_part = filename.split('-')[0]
            dates.append(pd.to_datetime(date_part, format='%Y%m%d'))
        except ValueError:
            print(f"Skipping file with invalid date format: {filename}")

    if not dates:
        raise ValueError("No valid date information found in filenames.")

    dates = pd.Series(dates)
    end_date = dates.max()
    start_date = end_date - pd.Timedelta(days=6)
    return start_date, end_date

def process_file(file_path):
    return pd.read_parquet(file_path)

def load_data_concurrent(filenames, directory):
    start_time = time.time()
    data_frames = []
    with ThreadPoolExecutor(max_workers=16) as executor:  # Increased concurrency
        future_to_file = {executor.submit(process_file, os.path.join(directory, file)): file for file in filenames if os.path.exists(os.path.join(directory, file))}
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                data_frames.append(future.result())
            except Exception as e:
                print(f"Exception occurred: {e}")
    data = pd.concat(data_frames)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds.")
    return data

directory = 'UREA_GABUNG'
start_date, end_date = get_date_range_from_directory(directory)
print("Start Date:", start_date)
print("End Date:", end_date)

date_range = pd.date_range(start=start_date, end=end_date, freq='D')
filenames = [f"{date.strftime('%Y%m%d')}-urea_aces_module.parquet" for date in date_range]
print("Filenames:", filenames)

data = load_data_concurrent(filenames, directory)
data['Time_Stamp'] = pd.to_datetime(data['Time_Stamp'], format='%Y-%m-%d %H:%M:%S')

for col in ['U2_62PIC101', 'U2_62FI155']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data.fillna(0, inplace=True)
data.set_index('Time_Stamp', inplace=True)
data.index = pd.to_datetime(data.index)

# Preprocessing data
start_time = time.time()
data_resampled = data[['U2_62PIC101', 'U2_62FI155']].resample('S').mean()
print(f"Data resampling completed in {time.time() - start_time:.2f} seconds.")

# Scale the data
start_time = time.time()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_resampled)
print(f"Data scaling completed in {time.time() - start_time:.2f} seconds.")

# Create sequences for LSTM
sequence_length = 60
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length])

X, y = np.array(X), np.array(y)
print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

# Splitting data into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Use tf.data.Dataset for better performance
train_dataset = Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)

# Building the LSTM model with more layers and improvements
start_time = time.time()
strategy = tf.distribute.MirroredStrategy()  # Use multiple CPUs

with strategy.scope():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Define the input shape here
    model.add(LSTM(units=120, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Additional LSTM layer
    model.add(LSTM(units=60, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Additional LSTM layer
    model.add(LSTM(units=60, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Dense layer with regularization
    model.add(Dense(units=60, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(Dense(units=2))  # Output layer for 2 variables

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)

    model.compile(optimizer='adam', loss='mean_squared_error')
print(f"Model building completed in {time.time() - start_time:.2f} seconds.")

# Train model with mini-batch and callbacks
start_time = time.time()
model.fit(train_dataset, epochs=100, verbose=1, callbacks=[lr_scheduler, early_stopping])  # Mini-batch training
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# Predict forecasting
start_time = time.time()
predictions = model.predict(test_dataset)
predictions = scaler.inverse_transform(predictions)  # Inverse scaling to get actual values
prediction_time = time.time() - start_time
print(f"Prediction completed in {prediction_time:.2f} seconds.")

# Calculate accuracy and error
mse = mean_squared_error(y_test, predictions)
accuracy = accuracy_score(np.round(y_test[:, 0]), np.round(predictions[:, 0])) * 100
print(f"Mean Squared Error: {mse:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

# Add forecast dataframe
forecast_dates = pd.date_range(start=data_resampled.index[-1], periods=len(predictions) + 1, freq='H')[1:]
forecast_df = pd.DataFrame({
    'Time_Stamp': forecast_dates,
    'U2_62FI155': np.round(predictions[:, 1], 2),
    'U2_62PIC101': np.round(predictions[:, 0], 2)
})

# Logic for shutdown and normal
shutdown_threshold_fi155 = 5
shutdown_threshold_pic101 = 150

forecast_df['shutdown'] = np.where(
    (forecast_df['U2_62FI155'] < shutdown_threshold_fi155 + 0.5) &
    (forecast_df['U2_62PIC101'] < shutdown_threshold_pic101 + 5),
    1, 0
)

forecast_df['status'] = forecast_df['shutdown'].map({1: 'Shutdown', 0: 'Normal'})
print(forecast_df.head(10))

# Save results
output_directory = 'hasil_prediksi_lstm'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
output_filename = f"{date_range_str}_prediksi_lstm.parquet"
output_file = os.path.join(output_directory, output_filename)

forecast_df.to_parquet(output_file, index=False)
print(f"Forecasting completed and results saved to {output_file}")
