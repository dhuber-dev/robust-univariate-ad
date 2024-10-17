import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Generate time series data
n = 200  # Total number of observations
t = np.arange(n)

# Stable period data (t = 0 to 49)
stable_period = 50
stable_mean = 0
stable_std = 1
stable_data = np.random.normal(loc=stable_mean, scale=stable_std, size=stable_period)

# Spike anomaly at t = 50
spike_value = 10  # Magnitude of the spike
spike_data = np.array([spike_value])

# Drift period data (t = 51 to 199)
drift_period = n - stable_period - 1
drift = np.linspace(0, 5, drift_period)  # Linear drift from 0 to 5
drift_data = np.random.normal(loc=drift, scale=stable_std)

# Combine all data into a single time series
X_t = np.concatenate([stable_data, spike_data, drift_data])

# Identify anomalies (here, the spike at t = 50)
anomaly_indices = [stable_period]  # Index of the anomaly
anomaly_values = X_t[anomaly_indices]  # Value(s) of the anomaly

# Create a DataFrame to facilitate plotting with Seaborn
data_type = ['Normal'] * n  # Initialize all data points as 'Normal'
for idx in anomaly_indices:
    data_type[idx] = 'Anomaly'  # Mark anomalies
df = pd.DataFrame({'Time': t, 'Observation': X_t, 'Type': data_type})

# Plot with consistent colors and scientific layout adjustments
plt.figure(figsize=(14, 6))

# Colors for the plot
normal_color = 'blue'
anomaly_color = 'red'

# Styling parameters for scientific papers
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = (4, 4)
rcParams['savefig.dpi'] = 300
rcParams['figure.dpi'] = 300

# ------------- Run chart (Time Series) -------------
plt.figure()
plt.plot(df['Time'],
         df['Observation'],
         marker='o',
         linestyle='-',
         markersize=3,
         color=normal_color,
         label='Normal Data')

# Highlight anomaly data point(s)
anomaly_data = df[df['Type'] == 'Anomaly']
plt.plot(anomaly_data['Time'],
         anomaly_data['Observation'],
         'o',
         markersize=8,
         color=anomaly_color,
         label='Anomaly')

plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Observation $X_t$', fontsize=12)
plt.legend(fontsize=10)
# plt.title("Time Series", fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('printed/run_chart_with_anomaly.png')
plt.close()

# ------------- Distribution chart -------------
plt.figure()

dist = sns.histplot(
    data=df,
    kde=True,
    x='Observation',
    hue='Type',
    bins=30,
    multiple='stack',
    edgecolor='black',
    palette={'Normal': normal_color, 'Anomaly': anomaly_color}
)

plt.xlabel('Observation $X_t$', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
# plt.title("Distribution", fontsize=14)
dist.legend_.set_title(None)
plt.tight_layout()
plt.savefig('printed/distribution_chart_with_anomaly.png')
plt.close()

# Adjust layout and display plots
plt.tight_layout()
plt.show()
