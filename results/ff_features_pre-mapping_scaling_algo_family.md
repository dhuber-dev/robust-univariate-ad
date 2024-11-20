## Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| distance | 0.6937 | 0.9778 | 0.8116 | 857 |
| encoding | 0.8116 | 0.3613 | 0.5000 | 155 |
| forecasting | 0.9762 | 0.6244 | 0.7616 | 197 |
| trees | 0.2222 | 0.0099 | 0.0189 | 203 |
| Accuracy | - | - | 0.7217 | 1412 |
| Macro Avg | 0.6759 | 0.4933 | 0.5230 | 1412 |
| Weighted Avg | 0.6783 | 0.7217 | 0.6565 | 1412 |

Input Type: features
Learning Rate: 0.001
Batch Size: 64
Number of Epochs: 200
Mapping: {np.int64(0): 'distance', np.int64(1): 'encoding', np.int64(2): 'forecasting', np.int64(3): 'trees'}