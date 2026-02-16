import pandas as pd
import matplotlib.pyplot as plt

# Data
data = {
    'Epoch': list(range(1, 21)),
    'Loss_MobileNetV2': [2.2691, 2.0511, 1.5577, 1.1592, 0.9021, 0.7101, 0.5866, 0.4987, 0.4285, 0.3745,
                         0.3294, 0.2874, 0.2508, 0.219, 0.1862, 0.1602, 0.1344, 0.1154, 0.103, 0.0925],
    'Loss_EfficientNetV2L': [1.7815, 1.3627, 0.8553, 0.5996, 0.4654, 0.3671, 0.2894, 0.2194, 0.1709, 0.1296,
                              0.1021, 0.0793, 0.0652, 0.0616, 0.0539, 0.0449, 0.0355, 0.0261, 0.022, 0.0183]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss_MobileNetV2'], label='Loss_MobileNetV2', marker='o', color='red')
plt.plot(df['Epoch'], df['Loss_EfficientNetV2L'], label='Loss_EfficientNetV2L', marker='o', color='blue')

# Title and labels
plt.title('Loss per Epoch (MobileNetV2 vs EfficientNetV2L)', fontdict={'fontname': 'Arial', 'fontsize': 16, 'fontweight': 'bold'})
plt.xlabel('Epoch', fontdict={'fontname': 'Arial', 'fontsize': 14, 'fontweight': 'bold'})
plt.ylabel('Loss', fontdict={'fontname': 'Arial', 'fontsize': 14, 'fontweight': 'bold'})
plt.xticks(df['Epoch'])

# Legend with font size via prop
plt.legend(loc='lower left', prop={'size': 12})

# Tick font size
plt.tick_params(axis='both', which='major', labelsize=12)

# Make grid lighter
plt.grid(True, alpha=0.5)

# Show plot
plt.tight_layout()
plt.show()
