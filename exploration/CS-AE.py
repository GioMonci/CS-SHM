"""
Super simple compressed sensing autoencoder example
- Fake 1D signal = two cosines added together
- Random sensing matrix compresses the signal
- Small neural network reconstructs the full signal
- Plot original vs reconstructed
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 1. Settings
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
signal_length = 128          # Full signal size
num_measurements = 8         # Compressed measurement size
train_samples = 3000
test_samples = 200
epochs = 200
batch_size = 64
learning_rate = 0.001
seed = 42

np.random.seed(seed)
torch.manual_seed(seed)

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 2. Make fake signal: two cosines added together
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def make_two_cosine_signal(signal_length=128):
    t = np.linspace(0, 1, signal_length, endpoint=False)

    amplitude_1 = np.random.uniform(0.5, 1.0)
    amplitude_2 = np.random.uniform(1.5, 2.7)

    frequency_1 = np.random.uniform(1.8, 3.0)
    frequency_2 = np.random.uniform(5.6, 12.6)

    phase_1 = np.random.uniform(0, 2 * np.pi)
    phase_2 = np.random.uniform(0, 2 * np.pi)

    signal = (
        amplitude_1 * np.cos(2.3 * np.pi * frequency_1 * t + phase_1) +
        amplitude_2 * np.cos(1.6 * np.pi * frequency_2 * t + phase_2)
    )

    # Normalize so signals stay in a similar range
    max_abs = np.max(np.abs(signal))
    if max_abs > 0:
        signal = signal / max_abs

    return signal.astype(np.float32)

def make_dataset(num_samples, signal_length=128):
    signals = [make_two_cosine_signal(signal_length) for _ in range(num_samples)]
    return np.stack(signals)

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 3. Build dataset
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
x_train = make_dataset(train_samples, signal_length)
x_test = make_dataset(test_samples, signal_length)

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 4. Random sensing matrix Phi
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Shape: (num_measurements, signal_length)
# This compresses a full signal x into y = Phi x
Phi = np.random.randn(num_measurements, signal_length).astype(np.float32)

# Normalize rows a bit so values don't get too wild
Phi = Phi / np.linalg.norm(Phi, axis=1, keepdims=True)

def measure_signal_batch(signal_batch, phi):
    """
    signal_batch shape: (batch_size, signal_length)
    phi shape:          (num_measurements, signal_length)

    Returns:
        measurements shape: (batch_size, num_measurements)
    """
    return signal_batch @ phi.T

y_train = measure_signal_batch(x_train, Phi)
y_test = measure_signal_batch(x_test, Phi)

# Convert to tensors
x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)

x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test)

train_dataset = torch.utils.data.TensorDataset(y_train_tensor, x_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 5. Decoder network
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Input  = compressed measurements y
# Output = reconstructed full signal x_hat
class CSDecoder(nn.Module):
    def __init__(self, measurement_dim, signal_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(measurement_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, signal_dim)
        )

    def forward(self, y):
        return self.net(y)

model = CSDecoder(num_measurements, signal_length)

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 6. Train
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for measurement_batch, original_batch in train_loader:
        optimizer.zero_grad()

        reconstructed_batch = model(measurement_batch)
        loss = criterion(reconstructed_batch, original_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{epochs}, Loss: {epoch_loss:.6f}")

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 7. Test on one fresh signal
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
model.eval()

original_signal = make_two_cosine_signal(signal_length)

# Take compressed measurements y = Phi x
measurement_vector = original_signal @ Phi.T

original_tensor = torch.tensor(original_signal).unsqueeze(0)
measurement_tensor = torch.tensor(measurement_vector).unsqueeze(0)

with torch.no_grad():
    reconstructed_tensor = model(measurement_tensor)

reconstructed_signal = reconstructed_tensor.squeeze(0).numpy()

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 8. Plot original vs reconstructed
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
plt.figure(figsize=(10, 4))
plt.plot(original_signal, label="Original Signal")
plt.plot(reconstructed_signal, "--", label="Reconstructed Signal")
plt.title("Compressed Sensing with a Simple Neural Decoder")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 9. Plot training loss
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# 10. Print basic info
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
compression_ratio = signal_length / num_measurements
print(f"Signal length: {signal_length}")
print(f"Measurements: {num_measurements}")
print(f"Compression ratio: {compression_ratio:.2f}:1")