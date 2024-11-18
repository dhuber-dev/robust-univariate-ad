## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6074 | 1.0000 | 0.7557 | 857 |
| encoding | 0.0000 | 0.0000 | 0.0000 | 155 |
| forecasting | 1.0000 | 0.0051 | 0.0101 | 197 |
| trees | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.6076 | 1412 |
| Macro Avg | 0.4018 | 0.2513 | 0.1915 | 1412 |
| Weighted Avg | 0.5082 | 0.6076 | 0.4601 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}