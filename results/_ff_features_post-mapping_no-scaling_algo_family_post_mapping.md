## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6069 | 1.0000 | 0.7554 | 857 |
| encoding | 0.0000 | 0.0000 | 0.0000 | 155 |
| forecasting | 0.0000 | 0.0000 | 0.0000 | 197 |
| trees | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.6069 | 1412 |
| Macro Avg | 0.1517 | 0.2500 | 0.1888 | 1412 |
| Weighted Avg | 0.3684 | 0.6069 | 0.4585 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}