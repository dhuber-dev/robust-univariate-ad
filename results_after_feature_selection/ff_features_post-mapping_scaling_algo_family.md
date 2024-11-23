## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.8348 | 0.7666 | 0.7993 | 857 |
| encoding | 0.8118 | 0.4452 | 0.5750 | 155 |
| forecasting | 0.7789 | 0.7513 | 0.7649 | 197 |
| trees | 0.3286 | 0.5665 | 0.4159 | 203 |
| Accuracy | - | - | 0.7004 | 1412 |
| Macro Avg | 0.6885 | 0.6324 | 0.6388 | 1412 |
| Weighted Avg | 0.7517 | 0.7004 | 0.7147 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}