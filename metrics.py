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