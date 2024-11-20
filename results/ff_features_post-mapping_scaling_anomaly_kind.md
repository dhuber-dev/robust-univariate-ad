## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.2254 | 0.1185 | 0.1553 | 135 |
| extremum | 0.8466 | 0.7005 | 0.7667 | 197 |
| frequency | 0.1438 | 0.2391 | 0.1796 | 92 |
| mean | 0.3247 | 0.6900 | 0.4416 | 200 |
| pattern | 0.1718 | 0.2593 | 0.2066 | 108 |
| pattern-shift | 0.2018 | 0.1983 | 0.2000 | 116 |
| platform | 0.5364 | 0.3932 | 0.4538 | 206 |
| trend | 0.7386 | 0.4194 | 0.5350 | 155 |
| variance | 0.1310 | 0.0542 | 0.0767 | 203 |
| Accuracy | - | - | 0.3697 | 1412 |
| Macro Avg | 0.3689 | 0.3414 | 0.3350 | 1412 |
| Weighted Avg | 0.4029 | 0.3697 | 0.3643 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}