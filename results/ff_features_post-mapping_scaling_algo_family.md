## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.7084 | 0.8903 | 0.7890 | 857 |
| encoding | 0.7386 | 0.4194 | 0.5350 | 155 |
| forecasting | 0.8466 | 0.7005 | 0.7667 | 197 |
| trees | 0.1310 | 0.0542 | 0.0767 | 203 |
| Accuracy | - | - | 0.6919 | 1412 |
| Macro Avg | 0.6062 | 0.5161 | 0.5418 | 1412 |
| Weighted Avg | 0.6480 | 0.6919 | 0.6556 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}