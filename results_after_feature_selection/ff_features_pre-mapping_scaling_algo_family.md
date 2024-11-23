## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.7154 | 0.9708 | 0.8238 | 857 |
| encoding | 0.8276 | 0.4645 | 0.5950 | 155 |
| forecasting | 0.9103 | 0.7208 | 0.8045 | 197 |
| trees | 0.1667 | 0.0049 | 0.0096 | 203 |
| Accuracy | - | - | 0.7415 | 1412 |
| Macro Avg | 0.6550 | 0.5403 | 0.5582 | 1412 |
| Weighted Avg | 0.6760 | 0.7415 | 0.6789 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}