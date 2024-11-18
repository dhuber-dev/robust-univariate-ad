## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.2812 | 0.0667 | 0.1078 | 135 |
| extremum | 0.8000 | 0.6701 | 0.7293 | 197 |
| frequency | 0.1556 | 0.0761 | 0.1022 | 92 |
| mean | 0.3370 | 0.6150 | 0.4354 | 200 |
| pattern | 0.1803 | 0.1019 | 0.1302 | 108 |
| pattern-shift | 0.1706 | 0.3103 | 0.2202 | 116 |
| platform | 0.3969 | 0.5049 | 0.4444 | 206 |
| trend | 0.6889 | 0.4000 | 0.5061 | 155 |
| variance | 0.1657 | 0.1478 | 0.1562 | 203 |
| Accuracy | - | - | 0.3640 | 1412 |
| Macro Avg | 0.3529 | 0.3214 | 0.3146 | 1412 |
| Weighted Avg | 0.3815 | 0.3640 | 0.3513 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}