# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:11:15 2019

@author: shikhar
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def csv_import(name, sep, header):
    """
    This function takes as input the name of csv file to be imported along with the separator and returns a dataframe
    :param name: name of csv file to be imported
    :param sep: separator value 
    :param header: header row 
    :returns: data frame of CSV file 
    """
    csv_file = pd.read_csv(name, sep = sep, header = header) ##loading data using read_csv from pandas
    return csv_file #returning the data structure
           
  
def select_cols(df,list_col):
    """
    This function takes as input the name of a dataframe and a list of column and it returns the dataframe with just those columns passed
    :param df: dataframe to which the columns is selected
    :param list_col: list of columns that you need in the new dataframe 
    :returns: data frame subset based on the columns passed  
    """
    df = df[list_col] ##loading data using read_csv from pandas
    return df #returning the data structure           


def filter_data(df, column_name,column_value,index_drop):
    """
    This function takes as input the name of a dataframe, column name, and a column value to which we need the filter 
    and it returns the dataframe with filter applied
    :param df: dataframe to which the columns is selected
    :param column_name: name of the column to which filter is to be applied
    :param column_value: value of the column which is needed in the output
    :param index_drop: drop index either True or False
    :returns: data frame subset based on column name and value passed
    """
    df = df.loc[df[column_name] == column_value].reset_index(drop = index_drop)
    return df #returning the data structure       

def LinearRegressionM(X_train, X_test, y_train, y_test):
    """
    This function takes as input our training and testing predictor and outcome variables and performs linear regression and returns
    a data frame having actual and predicted value
    :param X_train: data of training having all predictor variables
    :param X_test: data of testing having all predictor variables
    :param y_train: data of training having our outcome variable
    :param y_test: data of testing having our outcome variable
    :returns: data frame with actual and predicted value
    """
    lin_reg_mod = LinearRegression()
    lin_reg_mod.fit(X_train, y_train)
    pred = lin_reg_mod.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
    return df #returning the data structure


def vif_table(X):
    """
    Variance inflation factors for the predictor columns of X.

    :param X: data frame of predictor variables (numeric)
    :returns: data frame with one row per predictor and its VIF.
              VIF above 5-10 is usually read as strong collinearity.
    """
    X_const = add_constant(X.astype(float))
    vif = pd.DataFrame({
        'Variable': X_const.columns,
        'VIF': [variance_inflation_factor(X_const.values, i)
                for i in range(X_const.shape[1])],
    })
    return vif[vif['Variable'] != 'const'].reset_index(drop=True)


def repeated_cv_mae(X, y_share, housing_units, n_splits=5, n_repeats=10,
                    random_state=42):
    """
    Repeated K-fold cross-validation for the broadband model.

    The model predicts the *share* of housing units with fiber
    (y_share = fiber units / housing units). Count predictions are the
    predicted share times the county's housing units, clipped to
    [0, housing units], so a prediction can never be negative or exceed
    the number of housing units.

    The baseline predicts every county with the training folds' mean
    fiber share times that county's housing units.

    :param X: predictor data frame
    :param y_share: fiber share per row (0 to 1)
    :param housing_units: housing-unit count per row (same order as X)
    :returns: (fold-level results data frame, summary dict of
              mean/SD MAE for the model and the baseline)
    """
    y_share = np.asarray(y_share, dtype=float)
    y_count = y_share * np.asarray(housing_units, dtype=float)
    housing_units = np.asarray(housing_units, dtype=float)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        model = LinearRegression()
        model.fit(X.iloc[train_idx], y_share[train_idx])
        pred_share = np.clip(model.predict(X.iloc[test_idx]), 0, 1)
        pred_count = np.clip(pred_share * housing_units[test_idx],
                             0, housing_units[test_idx])
        baseline_count = np.clip(y_share[train_idx].mean()
                                 * housing_units[test_idx],
                                 0, housing_units[test_idx])
        rows.append({
            'fold': fold,
            'model_mae': mean_absolute_error(y_count[test_idx], pred_count),
            'baseline_mae': mean_absolute_error(y_count[test_idx],
                                                baseline_count),
            'share_mae': mean_absolute_error(y_share[test_idx], pred_share),
        })
    folds = pd.DataFrame(rows)
    summary = {
        'n_folds': len(folds),
        'model_mae_mean': folds['model_mae'].mean(),
        'model_mae_sd': folds['model_mae'].std(ddof=1),
        'baseline_mae_mean': folds['baseline_mae'].mean(),
        'baseline_mae_sd': folds['baseline_mae'].std(ddof=1),
        'share_mae_mean': folds['share_mae'].mean(),
        'share_mae_sd': folds['share_mae'].std(ddof=1),
    }
    return folds, summary


