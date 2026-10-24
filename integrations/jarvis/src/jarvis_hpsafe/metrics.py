"""JARVIS metrics: probabilities at threshold 0.5 for ACC, raw units for MAE."""
import numpy as np

def score_predictions(task_type,y,prediction):
    y=np.asarray(y);prediction=np.asarray(prediction)
    assert np.isfinite(prediction).all()
    return float(np.mean((prediction>.5)==y)) if task_type=='classification' else float(np.mean(np.abs(prediction-y)))
