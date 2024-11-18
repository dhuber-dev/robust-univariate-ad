## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.5644 | 0.6698 | 0.6126 | 857 |
| encoding | 0.0000 | 0.0000 | 0.0000 | 155 |
| forecasting | 0.2000 | 0.0203 | 0.0369 | 197 |
| trees | 0.1387 | 0.2562 | 0.1799 | 203 |
| Accuracy | - | - | 0.4462 | 1412 |
| Macro Avg | 0.2258 | 0.2366 | 0.2073 | 1412 |
| Weighted Avg | 0.3904 | 0.4462 | 0.4028 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}