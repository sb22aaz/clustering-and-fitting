#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import numpy as np

# =============================================================================
# return_df method for returen the data frames
# =============================================================================
def return_df(fname):
    '''
    this function takes a file name as a parameter and returns two data frames
    one is of countries as columns and one as years as columns

    Parameters
    ----------
    fname : String
        file name

    Returns
    -------
    df : DataFrame
        DataFrame with years as columns.
    dfT : DtaFrame
        DataFrame with countries as columns.

    '''
    data = pd.read_csv(fname)
    df=pd.DataFrame(data)
    dataT = pd.read_csv(fname,header=None,index_col=0).T
    dfT = pd.DataFrame(dataT)
    dfT=dfT.rename(columns={"Country Name":"Year"})
    return df,dfT

# =============================================================================
# convert the non numbers of the Dataframe to numbers
# =============================================================================
def converttonumbers(df,columnlist):
    '''
    In this method we will convert the non numbers to numbers

    Parameters
    ----------
    df : DataFrame
        dataframe that need to update.
    columnlist : list
        A list of columns need to be converted.

    Returns
    -------
    df : DataFrame
        Updated DataFrame in which updated columns to numeric type.

    '''
    df = df.replace(to_replace=".." , value="0")
    df[columnlist] = df[columnlist].apply(pd.to_numeric)
    return df