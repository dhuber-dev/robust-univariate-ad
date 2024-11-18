## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.0000 | 0.0000 | 0.0000 | 135 |
| extremum | 0.2012 | 0.1675 | 0.1828 | 197 |
| frequency | 0.0000 | 0.0000 | 0.0000 | 92 |
| mean | 0.0000 | 0.0000 | 0.0000 | 200 |
| pattern | 0.0000 | 0.0000 | 0.0000 | 108 |
| pattern-shift | 0.0000 | 0.0000 | 0.0000 | 116 |
| platform | 0.1398 | 0.8447 | 0.2398 | 206 |
| trend | 0.0000 | 0.0000 | 0.0000 | 155 |
| variance | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.1466 | 1412 |
| Macro Avg | 0.0379 | 0.1125 | 0.0470 | 1412 |
| Weighted Avg | 0.0485 | 0.1466 | 0.0605 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}