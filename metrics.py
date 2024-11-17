def get_precision(y_true, y_pred):
    tp = 0
    fp = 0
    for actual, pred in zip(y_true, y_pred):
      if pred=="Not the A-hole":
        if actual=="Not the A-hole":
          tp+=1
        else:
          fp+=1

    return tp / (tp+fp) if tp+fp != 0 else 0

def get_recall(y_true, y_pred):
    tp = 0
    fn = 0
    for actual, pred in zip(y_true, y_pred):
      if pred=="Not the A-hole":
        if actual=="Not the A-hole":
          tp+=1
        else:
          fn+=1

    return tp / (tp+fn) if tp+fn != 0 else 0

def get_fscore(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    return 2 * (precision * recall / (precision + recall)) if precision + recall != 0 else 0