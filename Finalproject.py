# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:11:15 2019

@author: shikhar-pragya-mohita
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

def csv_import(name, sep, header):
    """
    This function takes as input the name of csv file to be imported along with the seperator and returns a dataframe
    :param name: name of csv file to be imported
    :param sep: seperator value 
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
    This function takes as input the name of a dataframe, column name, and a column value to which we need the fileter 
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

 

