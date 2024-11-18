## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.1448 | 0.1556 | 0.1500 | 135 |
| extremum | 0.2000 | 0.0203 | 0.0369 | 197 |
| frequency | 0.0000 | 0.0000 | 0.0000 | 92 |
| mean | 0.2234 | 0.7250 | 0.3416 | 200 |
| pattern | 0.0000 | 0.0000 | 0.0000 | 108 |
| pattern-shift | 0.0000 | 0.0000 | 0.0000 | 116 |
| platform | 0.1773 | 0.1893 | 0.1831 | 206 |
| trend | 0.0000 | 0.0000 | 0.0000 | 155 |
| variance | 0.1387 | 0.2562 | 0.1799 | 203 |
| Accuracy | - | - | 0.1848 | 1412 |
| Macro Avg | 0.0982 | 0.1496 | 0.0991 | 1412 |
| Weighted Avg | 0.1192 | 0.1848 | 0.1204 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}