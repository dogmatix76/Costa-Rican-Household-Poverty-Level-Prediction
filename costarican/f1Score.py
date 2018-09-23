# custom metric
from sklearn.metrics import f1_score
import numpy as np
def macro_f1_score(labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)

    metric_value = f1_score(labels, predictions, average = 'macro')

    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True
