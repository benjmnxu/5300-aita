# Scoring Script Documentation

This document explains the evaluation metrics implemented in the script `score.py` and provides instructions for running the script from the command line.

## Metrics

The script calculates the following metrics for multi-class classification:

### 1. **Accuracy**

- **Definition**: The proportion of correctly classified instances out of the total number of instances.
- **Formula**: Accuracy = Correct Predictions / Total Predictions

### 2. **Weighted Metrics**

These metrics take class imbalance into account by weighting each class’s contribution based on the number of true instances of that class:

- **Weighted Precision**
- **Weighted Recall**
- **Weighted F1-Score**

### 3. **Macro Metrics**

These metrics treat all classes equally, regardless of class frequency:

- **Macro Precision**
- **Macro Recall**
- **Macro F1-Score**

### 4. **Per-Class Metrics**

For each class, the following metrics are calculated:

- **Precision**: proportion of correct predictions among all predictions for a class.
- **Recall**: proportion of correctly predicted instances among all actual instances of a class.
- **F1-Score**: harmonic mean of precision and recall.

## How to Run the Script

### Input

The script requires two files:

1. `true_labels.txt`: File containing the ground truth labels, one label per line.
2. `predicted_labels.txt`: File containing the predicted labels, one label per line.

### Command-Line Usage

Run the script from the command line as follows:

```
python score.py <true_labels_file> <predicted_labels_file>
```

### Example Output

- Accuracy: 0.5000
- Weighted Precision: 0.5833
- Weighted Recall: 0.5000
- Weighted F1-Score: 0.4667
- Macro Precision: 0.6250
- Macro Recall: 0.5833
- Macro F1-Score: 0.5667

Per-Class Metrics:

- Class: NTA - Precision: 0.6667, Recall: 0.6667, F1-Score: 0.6667
- Class: YTA - Precision: 0.5000, Recall: 0.5000, F1-Score: 0.5000
- Class: ESH - Precision: 0.5000, Recall: 0.5000, F1-Score: 0.5000
- Class: NAH - Precision: 1.0000, Recall: 1.0000, F1-Score: 1.0000
