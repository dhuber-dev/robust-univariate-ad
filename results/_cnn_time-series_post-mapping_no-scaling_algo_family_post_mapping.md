## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6074 | 1.0000 | 0.7557 | 857 |
| encoding | 0.0000 | 0.0000 | 0.0000 | 155 |
| forecasting | 1.0000 | 0.0051 | 0.0101 | 197 |
| trees | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.6076 | 1412 |
| Macro Avg | 0.4018 | 0.2513 | 0.1915 | 1412 |
| Weighted Avg | 0.5082 | 0.6076 | 0.4601 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}