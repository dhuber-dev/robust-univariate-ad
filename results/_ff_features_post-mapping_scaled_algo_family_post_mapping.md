## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.7131 | 0.8121 | 0.7594 | 857 |
| encoding | 0.6889 | 0.4000 | 0.5061 | 155 |
| forecasting | 0.8000 | 0.6701 | 0.7293 | 197 |
| trees | 0.1657 | 0.1478 | 0.1562 | 203 |
| Accuracy | - | - | 0.6516 | 1412 |
| Macro Avg | 0.5919 | 0.5075 | 0.5378 | 1412 |
| Weighted Avg | 0.6439 | 0.6516 | 0.6407 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}