## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.1440 | 0.5407 | 0.2274 | 135 |
| extremum | 0.3043 | 0.2843 | 0.2940 | 197 |
| frequency | 0.0000 | 0.0000 | 0.0000 | 92 |
| mean | 0.2285 | 0.3850 | 0.2868 | 200 |
| pattern | 0.0000 | 0.0000 | 0.0000 | 108 |
| pattern-shift | 0.0820 | 0.1810 | 0.1129 | 116 |
| platform | 0.8000 | 0.1748 | 0.2869 | 206 |
| trend | 0.9524 | 0.1290 | 0.2273 | 155 |
| variance | 0.1311 | 0.0394 | 0.0606 | 203 |
| Accuracy | - | - | 0.2061 | 1412 |
| Macro Avg | 0.2936 | 0.1927 | 0.1662 | 1412 |
| Weighted Avg | 0.3354 | 0.2061 | 0.1882 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}