## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6113 | 1.0000 | 0.7587 | 857 |
| encoding | 1.0000 | 0.0194 | 0.0380 | 155 |
| forecasting | 0.0000 | 0.0000 | 0.0000 | 197 |
| trees | 0.0000 | 0.0000 | 0.0000 | 203 |
| Accuracy | - | - | 0.6091 | 1412 |
| Macro Avg | 0.4028 | 0.2548 | 0.1992 | 1412 |
| Weighted Avg | 0.4808 | 0.6091 | 0.4647 | 1412 |

Input Type: time-series
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}