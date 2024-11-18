## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.1381 | 0.9778 | 0.2420 | 135 |
| extremum | 0.7586 | 0.1117 | 0.1947 | 197 |
| frequency | 0.0000 | 0.0000 | 0.0000 | 92 |
| mean | 0.0000 | 0.0000 | 0.0000 | 200 |
| pattern | 0.0000 | 0.0000 | 0.0000 | 108 |
| pattern-shift | 0.0000 | 0.0000 | 0.0000 | 116 |
| platform | 0.2557 | 0.3835 | 0.3068 | 206 |
| trend | 0.0000 | 0.0000 | 0.0000 | 155 |
| variance | 0.3333 | 0.1921 | 0.2437 | 203 |
| Accuracy | - | - | 0.1926 | 1412 |
| Macro Avg | 0.1651 | 0.1850 | 0.1097 | 1412 |
| Weighted Avg | 0.2043 | 0.1926 | 0.1301 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}