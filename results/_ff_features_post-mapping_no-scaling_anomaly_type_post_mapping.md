## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.0000 | 0.0000 | 0.0000 | 135 |
| extremum | 0.0000 | 0.0000 | 0.0000 | 197 |
| frequency | 0.0000 | 0.0000 | 0.0000 | 92 |
| mean | 0.0000 | 0.0000 | 0.0000 | 200 |
| pattern | 0.0000 | 0.0000 | 0.0000 | 108 |
| pattern-shift | 0.0000 | 0.0000 | 0.0000 | 116 |
| platform | 0.1459 | 1.0000 | 0.2546 | 206 |
| trend | 0.0000 | 0.0000 | 0.0000 | 155 |
| variance | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.1459 | 1412 |
| Macro Avg | 0.0162 | 0.1111 | 0.0283 | 1412 |
| Weighted Avg | 0.0213 | 0.1459 | 0.0371 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}