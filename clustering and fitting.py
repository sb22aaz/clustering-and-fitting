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

def curveFit(dft,df1t,country):
    '''
    generating the curve fit for a given country

    Parameters
    ----------
    dft : DataFrame
        countries as columns dataframe.
    df1t : DataFrame
        Countries as columns dataframe.
    country : String
        countryname to plot curve fit.
    

    Returns
    -------
    None.

    '''
    countryNew = "New"+country
    dft=converttonumbers(dft, [country])
    df1t=converttonumbers(df1t, [country])
    popt, covar = opt.curve_fit(objective,dft[country],df1t[country])
    df1t[countryNew] = objective(dft[country], *popt)
    plt.figure()
    plt.plot(dft[country], df1t[country], color='red')
    plt.plot(dft[country], df1t[countryNew], color='blue')
    ci = np.std(df1t[country])/np.sqrt(len(df1t[country]))
    plt.fill_between(dft[country], (df1t[country]-ci), (df1t[country]+ci),
                 color='b', alpha=0.1)
    # setting x-axis label
    plt.xlabel(labels[0])
    # setting y-axis label
    plt.ylabel(labels[1])
    # creating the legend and it's position
    plt.legend(["Expected","Predicted"],loc = "upper right")
    #Mentioning the title of the plot
    plt.title("Curve Fit of India")
    # showing the plot 
    plt.show()

