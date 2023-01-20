#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import numpy as np


def lineplot_usingDF(df,countries,st):
    plt.figure()
    df.plot("Year",countries)
    plt.ylabel(st)
    plt.legend(countries,loc = "upper right")
    plt.show()
    
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
    dfT = dfT.rename(columns={"Country Name":"Year"})
    return df,dfT

# =============================================================================
# convert the non numbers of the Dataframe to numbers
# =============================================================================
def converttonumbers(df, columnlist):
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
    df = df.replace(to_replace = ".." , value = "0")
    df[columnlist] = df[columnlist].apply(pd.to_numeric)
    return df

# =============================================================================
# 
# =============================================================================
def objective(x, a, b, c):
    '''
    This is the function which returns the quadratic function
    Parameters
    ----------
    x : list
        Dependent variable for creating the equation.
    a : float
         coefficient of x.
    b : float
        coefficient of x2.
    c : float
        constant.

    Returns
    -------
    pandas.core.series.Series
        returns the quadratic function.

    '''
    return (a * x) + (b * x**2) + c

def curveFit(dft, df1t, country):
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
    dft = converttonumbers(dft, [country])
    df1t = converttonumbers(df1t, [country])
    popt, covar = opt.curve_fit(objective, dft[country], df1t[country])
    #predicting the respective Annual freshwater withdrawals for the population
    df1t[countryNew] = objective(dft[country], *popt)
    plt.figure()
    #plotting the original data
    plt.plot(dft[country], df1t[country], color = 'red')
    #plotting the predicted data
    plt.plot(dft[country], df1t[countryNew], color = 'blue')
    #generating the confidence ranges
    cr = np.std(df1t[country]) / np.sqrt(len(df1t[country]))
    #plotting confidence ranges
    plt.fill_between(dft[country], (df1t[country]-cr), (df1t[country]+cr),
                 color = 'b', alpha = 0.1)
    # setting x-axis label
    plt.xlabel(labels[0])
    # setting y-axis label
    plt.ylabel(labels[1])
    # creating the legend and it's position
    plt.legend(["Expected", "Predicted"], loc = "upper right")
    #Mentioning the title of the plot
    plt.title("Curve Fit of "+country) 
    # showing the plot 
    plt.show()


def kMeansCluster(df_fit, noclusters, label):
    '''
    Generating the k-means clustering for the given data

    Parameters
    ----------
    df_fit : DataFrame
        dataframe with which one uses k-means.
    noclusters : int
        number of clusters.
    label : list
        list of labels.

    Returns
    -------
    None.

    '''
    # creating KMeans with 4 clusters
    kmeans = cluster.KMeans(n_clusters = noclusters)
    # fitting the data
    kmeans.fit(df_fit) 
    # getting labels
    labels = kmeans.labels_
    # finding centers
    cen = kmeans.cluster_centers_
    # plotting the figure
    plt.figure(figsize=(6.0, 6.0))

    # Individual colours can be assigned to symbols. The label l is used to the 
    # select l-th number from the colour table.
    plt.scatter(df_fit["population"], df_fit["water"], c=labels, cmap="Accent")

    # colour map Accent selected to increase contrast between colours
    # show cluster centres
    for ic in range(noclusters):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize = 10)

    # setting x-axis label
    plt.xlabel(label[0])
    # setting y-axis label
    plt.ylabel(label[1])
    #Mentioning the title of the plot
    plt.title("K-means Clustering")
    plt.show()

# assigning the excel names
fname1 = "Annual freshwater withdrawals.csv"
fname2 = "Population growth.csv"

# invoking the function to get the data
populationDF,dft = return_df(fname2)
waterDF,df1t = return_df(fname1)

# taking the required years for plotting into an array
years = ["1990","2000","2012","2013","2014","2015","2016","2017","2018"]

# invoking convertonumbers methods to change the data to numeric form
df = converttonumbers(populationDF, years)
df1 = converttonumbers(waterDF, years)



country = "India"
#curve_fit(dft,df1t,country,countryNew)
labels = ["Population growth","Annual freshwater withdrawals"]
# =============================================================================
# applying the curve fit method to the data
# =============================================================================
curveFit(dft,df1t,country)
years = ["2019"]
# invoking convertonumbers methods to change the data of 2019 to numeric form
df = converttonumbers(populationDF, years)
df1 = converttonumbers(waterDF, years)
x = df["2019"]
y = df1["2019"]
#using curve fit
popt, covar = curve_fit(objective, x, y)
x.name = "population"
y.name = "water"
# using pandas series merge()
df_fit = pd.merge(x, y, right_index = True,
               left_index = True)
# =============================================================================
# applying clustring  for 2019 data
# =============================================================================
kMeansCluster(df_fit, 4, labels)