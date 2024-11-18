## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6191 | 0.9977 | 0.7641 | 857 |
| encoding | 0.0000 | 0.0000 | 0.0000 | 155 |
| forecasting | 0.8387 | 0.1320 | 0.2281 | 197 |
| trees | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.6239 | 1412 |
| Macro Avg | 0.3645 | 0.2824 | 0.2480 | 1412 |
| Weighted Avg | 0.4928 | 0.6239 | 0.4956 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}