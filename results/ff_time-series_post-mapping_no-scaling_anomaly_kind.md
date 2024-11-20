## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.0000 | 0.0000 | 0.0000 | 135 |
| extremum | 0.1265 | 0.3655 | 0.1880 | 197 |
| frequency | 0.0000 | 0.0000 | 0.0000 | 92 |
| mean | 0.5435 | 0.2500 | 0.3425 | 200 |
| pattern | 0.0952 | 0.0370 | 0.0533 | 108 |
| pattern-shift | 0.1963 | 0.2759 | 0.2294 | 116 |
| platform | 0.1534 | 0.4029 | 0.2222 | 206 |
| trend | 0.0000 | 0.0000 | 0.0000 | 155 |
| variance | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.1707 | 1412 |
| Macro Avg | 0.1239 | 0.1479 | 0.1150 | 1412 |
| Weighted Avg | 0.1404 | 0.1707 | 0.1301 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}