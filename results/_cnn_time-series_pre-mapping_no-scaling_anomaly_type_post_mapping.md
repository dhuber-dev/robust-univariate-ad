## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6269 | 0.9860 | 0.7664 | 857 |
| encoding | 1.0000 | 0.0323 | 0.0625 | 155 |
| forecasting | 0.6897 | 0.2030 | 0.3137 | 197 |
| trees | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.6303 | 1412 |
| Macro Avg | 0.5791 | 0.3053 | 0.2857 | 1412 |
| Weighted Avg | 0.5865 | 0.6303 | 0.5158 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}