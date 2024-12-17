import sys
from collections import Counter

'''
Classes: NTA, YTA, ESH, NAH
'''

def get_accuracy(y_true, y_pred):
  correct = sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred)
  return correct / len(y_true) if len(y_true) != 0 else 0

def get_weighted_precision(y_true, y_pred):
  return get_weighted_metric(get_precision, y_true, y_pred)

def get_weighted_recall(y_true, y_pred):
  return get_weighted_metric(get_recall, y_true, y_pred)

def get_weighted_fscore(y_true, y_pred):
  return get_weighted_metric(get_fscore, y_true, y_pred)

def get_weighted_metric(metric_fn, y_true, y_pred):
  '''
    Computes metric for each class, weighted by the number of true instances of each class.
  '''
  unique_labels = set(y_true + y_pred)
  label_counts = Counter(y_true)
  total_count = len(y_true)
  weighted_score = sum(
      (label_counts[label] / total_count) * metric_fn(y_true, y_pred, label)
      for label in unique_labels
  )
  return weighted_score

def get_macro_precision(y_true, y_pred):
  return get_macro_metric(get_precision, y_true, y_pred)

def get_macro_recall(y_true, y_pred):
  return get_macro_metric(get_recall, y_true, y_pred)

def get_macro_fscore(y_true, y_pred):
  return get_macro_metric(get_fscore, y_true, y_pred)

def get_macro_metric(metric_fn, y_true, y_pred):
  '''
    Computes metric for each class and averages them without considering class frequency.
  '''
  unique_labels = set(y_true + y_pred)
  scores = [metric_fn(y_true, y_pred, label) for label in unique_labels]
  return sum(scores) / len(unique_labels) if len(unique_labels) != 0 else 0

'''
Functions per class
'''

def get_precision(y_true, y_pred, label):
  tp = 0
  fp = 0
  for actual, pred in zip(y_true, y_pred):
      if pred == label:
          if actual == label:
              tp += 1
          else:
              fp += 1
  return tp / (tp + fp) if tp + fp != 0 else 0

def get_recall(y_true, y_pred, label):
  tp = 0
  fn = 0
  for actual, pred in zip(y_true, y_pred):
      if actual == label:
          if pred == label:
              tp += 1
          else:
              fn += 1
  return tp / (tp + fn) if tp + fn != 0 else 0

def get_fscore(y_true, y_pred, label):
  precision = get_precision(y_true, y_pred, label)
  recall = get_recall(y_true, y_pred, label)
  return 2 * (precision * recall / (precision + recall)) if precision + recall != 0 else 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <true_labels_file> <predicted_labels_file>")
        sys.exit(1)

    true_labels_file = sys.argv[1]
    predicted_labels_file = sys.argv[2]

    # Load labels from files
    with open(true_labels_file, 'r') as f:
        y_true = f.read().strip().split("\n")
    with open(predicted_labels_file, 'r') as f:
        y_pred = f.read().strip().split("\n")

    if len(y_true) != len(y_pred):
        print(len(y_true), len(y_pred))
        print("Error: Mismatch in the number of true and predicted labels.")
        sys.exit(1)

    # Run metrics
    print(f"Accuracy: {get_accuracy(y_true, y_pred):.4f}")
    print(f"Weighted Precision: {get_weighted_precision(y_true, y_pred):.4f}")
    print(f"Weighted Recall: {get_weighted_recall(y_true, y_pred):.4f}")
    print(f"Weighted F1-Score: {get_weighted_fscore(y_true, y_pred):.4f}")
    print(f"Macro Precision: {get_macro_precision(y_true, y_pred):.4f}")
    print(f"Macro Recall: {get_macro_recall(y_true, y_pred):.4f}")
    print(f"Macro F1-Score: {get_macro_fscore(y_true, y_pred):.4f}")

    # Per-class metrics
    unique_labels = set(y_true + y_pred)
    print("\nPer-Class Metrics:")
    for label in unique_labels:
        precision = get_precision(y_true, y_pred, label)
        recall = get_recall(y_true, y_pred, label)
        fscore = get_fscore(y_true, y_pred, label)
        print(f"Class: {label} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")