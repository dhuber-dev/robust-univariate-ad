## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6993 | 0.9417 | 0.8026 | 857 |
| encoding | 0.8507 | 0.3677 | 0.5135 | 155 |
| forecasting | 0.8562 | 0.6954 | 0.7675 | 197 |
| trees | 0.1613 | 0.0246 | 0.0427 | 203 |
| Accuracy | - | - | 0.7125 | 1412 |
| Macro Avg | 0.6419 | 0.5074 | 0.5316 | 1412 |
| Weighted Avg | 0.6605 | 0.7125 | 0.6567 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}