## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| amplitude | 0.1856 | 0.3630 | 0.2456 | 135 |
| extremum | 0.7789 | 0.7513 | 0.7649 | 197 |
| frequency | 0.1831 | 0.1413 | 0.1595 | 92 |
| mean | 0.6000 | 0.2400 | 0.3429 | 200 |
| pattern | 0.1745 | 0.2407 | 0.2023 | 108 |
| pattern-shift | 0.2025 | 0.1379 | 0.1641 | 116 |
| platform | 0.5972 | 0.4175 | 0.4914 | 206 |
| trend | 0.8118 | 0.4452 | 0.5750 | 155 |
| variance | 0.3286 | 0.5665 | 0.4159 | 203 |
| Accuracy | - | - | 0.4037 | 1412 |
| Macro Avg | 0.4291 | 0.3670 | 0.3735 | 1412 |
| Weighted Avg | 0.4768 | 0.4037 | 0.4127 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'amplitude', np.int64(1): 'extremum', np.int64(2): 'frequency', np.int64(3): 'mean', np.int64(4): 'pattern', np.int64(5): 'pattern-shift', np.int64(6): 'platform', np.int64(7): 'trend', np.int64(8): 'variance'}