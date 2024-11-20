## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6632 | 0.8868 | 0.7589 | 857 |
| encoding | 0.9524 | 0.1290 | 0.2273 | 155 |
| forecasting | 0.3043 | 0.2843 | 0.2940 | 197 |
| trees | 0.1311 | 0.0394 | 0.0606 | 203 |
| Accuracy | - | - | 0.5977 | 1412 |
| Macro Avg | 0.5128 | 0.3349 | 0.3352 | 1412 |
| Weighted Avg | 0.5684 | 0.5977 | 0.5353 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}