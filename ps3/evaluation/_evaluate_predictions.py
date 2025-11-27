import numpy as np
import pandas as pd
from glum import TweedieDistribution
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_bias(pred_col, act_col, weight_col):
    pred_mean_claim = np.sum(pred_col*weight_col) / np.sum(weight_col)
    actual_mean_claim = np.sum(act_col*weight_col) / np.sum(weight_col)
    bias = pred_mean_claim - actual_mean_claim
    return bias

def compute_deviance(pred_col, act_col, weight_col):
    TweedieDist = TweedieDistribution(1.5)
    deviance = TweedieDist.deviance(act_col, pred_col, sample_weight=weight_col)
    return deviance

def compute_metrics(pred_col, act_col, weight_col, model):
    mae  = mean_absolute_error(act_col, pred_col, sample_weight = weight_col)
    rmse = mean_squared_error(act_col, pred_col, sample_weight = weight_col)
    deviance = compute_deviance(pred_col, act_col, weight_col)
    bias = compute_bias(pred_col, act_col, weight_col)
    df = pd.DataFrame()
    df["model"] = [model]
    df["mae"] = [mae]
    df["rmse"] = [rmse]
    df["deviance"] = [deviance]
    df["bias"] = [bias]
    return df
