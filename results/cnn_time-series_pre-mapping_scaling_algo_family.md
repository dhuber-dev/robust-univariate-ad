## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6208 | 0.9953 | 0.7647 | 857 |
| encoding | 0.0000 | 0.0000 | 0.0000 | 155 |
| forecasting | 0.8529 | 0.1472 | 0.2511 | 197 |
| trees | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.6246 | 1412 |
| Macro Avg | 0.3684 | 0.2856 | 0.2539 | 1412 |
| Weighted Avg | 0.4958 | 0.6246 | 0.4991 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}