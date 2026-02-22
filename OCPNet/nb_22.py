# Auto-generated from /data/private/user2/workspace/ocean/oneocean(iros-2026-code)/OCPNet/22.ipynb
# Generated at 2026-02-23 02:12:50
# Note: notebook outputs are omitted; IPython magics are commented out.

# %% [cell 1]
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

nc_file = "../MOPD_pipeline/Data/Combined/combined_environment.nc"

if not os.path.exists(nc_file):
    raise FileNotFoundError(f"File not found: {nc_file}")

dataset = xr.open_dataset(nc_file)

print("Variables in the dataset:")
print(dataset.variables.keys())

# statistics
def compute_stats(var):
    if var in ["time", "depth", "latitude", "longitude"]:
        return None  # Skip non-numeric variables
    
    data = dataset[var].values.flatten()
    data = data[~np.isnan(data)]  # Filter out NaN values
    stats = {
        "Min": np.min(data),
        "Max": np.max(data),
        "Mean": np.mean(data),
        "Median": np.median(data),
        "Std": np.std(data)
    }
    return stats

stats_dict = {}
for var in dataset.variables.keys():
    stats = compute_stats(var)
    if stats:
        stats_dict[var] = stats

print("\nStatistical Information:")
for var, stats in stats_dict.items():
    print(f"\nVariable: {var}")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

# Handle time variable
if "time" in dataset.variables:
    time_values = dataset["time"].values
    print("\nTime Range:")
    print(f"  Start Time: {time_values[0]}")
    print(f"  End Time: {time_values[-1]}")
    print(f"  Time Span: {time_values[-1] - time_values[0]}")

# %% [cell 2]
import xarray as xr
from visual import plot_3d_currents

nc_file = "../MOPD_pipeline/Data/Combined/combined_environment.nc"
dataset = xr.open_dataset(nc_file)

plot_3d_currents(dataset, output_dir="output", skip=20, arrow_size=0.05, arrow_height_offset=5, arrow_alpha=0.4, arrow_head_size=10)

# %% [cell 3]
import os
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

nc_file = "../MOPD_pipeline/Data/Combined/combined_environment.nc"
if not os.path.exists(nc_file):
    raise FileNotFoundError(f"File not found: {nc_file}")
dataset = xr.open_dataset(nc_file)
data_uo = np.squeeze(dataset['uo'].values, axis=1)
data_vo = np.squeeze(dataset['vo'].values, axis=1)
data_uo_norm = (data_uo - np.mean(data_uo)) / np.std(data_uo)
data_vo_norm = (data_vo - np.mean(data_vo)) / np.std(data_vo)
data_stack = np.stack((data_uo_norm, data_vo_norm), axis=-1)

test_mode = True
if test_mode:
    data_stack = data_stack[:20, :120, :120]
    timesteps = 3
else:
    timesteps = 20

X, y = [], []
for i in range(data_stack.shape[0] - timesteps):
    X.append(data_stack[i:i + timesteps])
    y.append(data_stack[i + timesteps])
X = np.array(X)
y = np.array(y)

split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

input_shape = (timesteps, X.shape[2], X.shape[3], X.shape[4])
model = Sequential([
    Input(shape=input_shape),
    ConvLSTM2D(8, (3, 3), padding='same', return_sequences=False, activation='relu'),
    BatchNormalization(),
    Conv2D(2, (3, 3), activation='linear', padding='same')
])
model.compile(optimizer='adam', loss='mse')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=1, verbose=1),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

epochs = 1 if test_mode else 10
batch_size = 1 if test_mode else 8
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('epoch')
plt.ylabel('mse loss')
plt.title('training history')
plt.legend()
plt.savefig('training_history.png')
plt.show()

sample_index = 0
pred = model.predict(X_val)
ground_truth = y_val[sample_index]
prediction = pred[sample_index]
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
channel_names = ['uo', 'vo']
for i in range(2):
    axes[i, 0].imshow(ground_truth[..., i], cmap='viridis')
    axes[i, 0].set_title(f'ground truth {channel_names[i]}')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(prediction[..., i], cmap='viridis')
    axes[i, 1].set_title(f'prediction {channel_names[i]}')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.savefig('prediction_result.png')
plt.show()

model.save('final_model.keras')

# %% [cell 4]
import os
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = '/GPU:0'
    except:
        device = '/CPU:0'
else:
    device = '/CPU:0'

nc_file = "../MOPD_pipeline/Data/Combined/combined_environment.nc"
if not os.path.exists(nc_file):
    raise FileNotFoundError(f"File not found: {nc_file}")
dataset = xr.open_dataset(nc_file)
data_uo = np.squeeze(dataset['uo'].values, axis=1)  # shape: (time, lat, lon)
data_vo = np.squeeze(dataset['vo'].values, axis=1)  # shape: (time, lat, lon)

# normalize
data_uo_norm = (data_uo - np.mean(data_uo)) / np.std(data_uo)
data_vo_norm = (data_vo - np.mean(data_vo)) / np.std(data_vo)
data_stack = np.stack((data_uo_norm, data_vo_norm), axis=-1)  # shape: (time, lat, lon, 2)

test_mode = False
if test_mode:
    data_stack = data_stack[:50]

# construct sliding window samples
timesteps = 10
X, y = [], []
for i in range(data_stack.shape[0] - timesteps):
    X.append(data_stack[i:i+timesteps])
    y.append(data_stack[i+timesteps])
X = np.array(X)  # shape: (samples, timesteps, lat, lon, channels)
y = np.array(y)  # shape: (samples, lat, lon, channels)

# split data
split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# build ConvLSTM
input_shape = (timesteps, X.shape[2], X.shape[3], X.shape[4])

with tf.device(device):
    model = Sequential([
        Input(shape=input_shape),
        ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True, activation='relu'),
        BatchNormalization(),
        ConvLSTM2D(32, (3, 3), padding='same', return_sequences=False, activation='relu'),
        BatchNormalization(),
        Conv2D(2, (3, 3), activation='linear', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse')

# set callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# training parameters
epochs = 10 if not test_mode else 2
batch_size = 8 if not test_mode else 4

# train the model
with tf.device(device):
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

# visualize training history and save figure
plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.savefig('training_history.png')  
plt.show()

sample_index = 0 
with tf.device(device):
    pred = model.predict(X_val)
# pred shape: (samples, lat, lon, 2)  choose sample_index sample
ground_truth = y_val[sample_index]
prediction = pred[sample_index]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
channel_names = ['uo', 'vo']
for i in range(2):
    axes[i, 0].imshow(ground_truth[..., i], cmap='viridis')
    axes[i, 0].set_title(f'ground truth {channel_names[i]}')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(prediction[..., i], cmap='viridis')
    axes[i, 1].set_title(f'prediction {channel_names[i]}')
    axes[i, 1].axis('off')
plt.tight_layout()
plt.savefig('prediction_result.png') 
plt.show()

# save the final model
model.save('final_model.keras')

