import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set random seed for reproducibility
np.random.seed(0)

# Time vector
t = np.arange(0, 100)

# 1. Stationary Time Series (White Noise)
y_stationary = np.random.normal(0, 1, size=len(t))

# 2. Non-Stationary Series (Time-Dependent Mean)
trend = t * 0.1  # Linear trend
y_nonstationary_mean = trend + np.random.normal(0, 1, size=len(t))

# 3. Non-Stationary Series (Time-Dependent Variance)
std_dev = 0.05 * t + 1  # Standard deviation increases over time
y_nonstationary_variance = np.random.normal(0, std_dev)

# 4. Non-Stationary Series (Time-Dependent Covariance)
phi = 0.8 * np.sin(0.1 * t)  # Time-varying AR(1) coefficient
y_nonstationary_covariance = np.zeros(len(t))
for i in range(1, len(t)):
    y_nonstationary_covariance[i] = phi[i] * y_nonstationary_covariance[i-1] + np.random.normal(0, 1)

# Styling parameters for scientific papers
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = (8, 4)
rcParams['savefig.dpi'] = 300
rcParams['figure.dpi'] = 300

# 1. Stationary Time Series
plt.figure()
plt.plot(t, y_stationary, linewidth=1)
# plt.title('Stationary Time Series', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('printed/stationary_time_series.png')
plt.close()

# 2. Non-Stationary Series (Time-Dependent Mean)
plt.figure()
plt.plot(t, y_nonstationary_mean, linewidth=1)
# plt.title('Non-Stationary Series (Time-Dependent Mean)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('printed/nonstationary_time_dependent_mean.png')
plt.close()

# 3. Non-Stationary Series (Time-Dependent Variance)
plt.figure()
plt.plot(t, y_nonstationary_variance, linewidth=1)
# plt.title('Non-Stationary Series (Time-Dependent Variance)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('printed/nonstationary_time_dependent_variance.png')
plt.close()

# 4. Non-Stationary Series (Time-Dependent Covariance)
plt.figure()
plt.plot(t, y_nonstationary_covariance, linewidth=1)
# plt.title('Non-Stationary Series (Time-Dependent Covariance)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('printed/nonstationary_time_dependent_covariance.png')
plt.close()
