# Scoring Script Documentation

This document explains the evaluation metrics implemented in the script `score.py` and provides instructions for running the script from the command line.

## Weighted F1-Score

The most suitable evaluation metric for our project is the weighted F1-score. It handles the imbalance in our data, as NTA is the most frequent class (over 64%) across our training, development, and test datasets, followed by YTA (about 20%), then NAH (about 9%), and lastly, ESH (about 5%).

Our project involves understanding community judgments ("NTA", "YTA", etc.), so capturing and balancing false positives and false negatives are both critical.

The weighted F1-score also is ideal for multi-class settings, as it aggregates per-class performance into a single, interpretable metric.

The **Weighted F1-Score** is calculated as the weighted average of the F1-Scores for each class, where the weight for each class is proportional to the number of true instances of that class.

#### Formula:

$$
\text{Weighted F1-Score} = \sum_{i=1}^{n} w_i \cdot \text{F1-Score}_i
$$

Where:

- $n$: Total number of classes (4).
- $ w_i $: Weight for class $i$, calculated as:
  $$
  w_i = \frac{\text{Number of True Instances of Class } i}{\text{Total Number of Instances}}
  $$
- $ \text{F1-Score}\_i $: F1-Score for class $i$, calculated as:
  $$
  \text{F1-Score}_i = \frac{2 \cdot \text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
  $$
- $\text{Precision}_i$: Precision for class $i$, defined as:
  $$
  \text{Precision}_i = \frac{\text{True Positives}_i}{\text{True Positives}_i + \text{False Positives}_i}
  $$
- $ \text{Recall}\_i$ Recall for class $i$, defined as:
  $$
  \text{Recall}_i = \frac{\text{True Positives}_i}{\text{True Positives}_i + \text{False Negatives}_i}
  $$

## All Metrics

Our `score.py` script calculates the various metrics for our multi-class classification:

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

For the majority class baseline, this is:
```
python score.py true_labels_dev.txt predicted_labels_simple.txt
```
and yields 0.6439 as the precision (dataset distribution) and 1.0 as the recall for the majority class - then 0 for everything else.

For the strong baseline, this is:
```
python score.py true_labels_dev.txt predicted_labels_strong.txt
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
