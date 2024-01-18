# -*- coding: utf-8 -*-

#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt

#Used from class
import cluster_tools as ct
import errors as err
import importlib


def reading_data(filepath):
    '''
    reading_data will create dataframe from file in the given filepath

    Parameters
    ----------
    filepath : STR
        File path or location.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame created from given filepath.

    '''
    df = pd.read_csv(filepath, skiprows=4)
    df = df.set_index('Country Name', drop=True)
    df = df.loc[:, '1960':'2021']

    return df


def transpose(df):
    '''
    transpose create transpose of given dataframe

    Parameters
    ----------
    df  : pandas.DataFrame
        DataFrame for which transpose to be found.

    Returns
    -------
    data_tr : pandas.DataFrame
        Transposed DataFrame of given DataFrame.

    '''
    df_tr = df.transpose()

    return df_tr


def correlation_and_scattermatrix(df):
    '''
    correlation_and_scattermatrix plots correlation matrix and scatter plots
    of data among columns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame for which analysis will be done.

    Returns
    -------
    None.

    '''
    corr = df.corr()
    print(corr)
    plt.figure(figsize=(10, 10))
    plt.matshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation between Years and Countries over Energy Use')
    plt.colorbar()
    plt.show()

    pd.plotting.scatter_matrix(df, figsize=(12, 12), s=5, alpha=0.8)
    plt.show()

    return


def cluster_number(df, df_normalised):
    '''
    cluster_number calculates the best number of clusters based on silhouette
    score

    Parameters
    ----------
    df : pandas.DataFrame
        Actual data.
    df_normalised : pandas.DataFrame
        Normalised data.

    Returns
    -------
    INT
        Best cluster number.

    '''

    clusters = []
    scores = []
    # loop over number of clusters
    for ncluster in range(2, 10):

        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Cluster fitting
        kmeans.fit(df_normalised)
        lab = kmeans.labels_

        # Silhoutte score over number of clusters
        print(ncluster, skmet.silhouette_score(df, lab))

        clusters.append(ncluster)
        scores.append(skmet.silhouette_score(df, lab))

    clusters = np.array(clusters)
    scores = np.array(scores)

    best_ncluster = clusters[scores == np.max(scores)]
    print()
    print('best n clusters', best_ncluster[0])

    return best_ncluster[0]


def clusters_and_centers(df, ncluster, y1, y2):
    '''
    clusters_and_centers will plot clusters and its centers for given data

    Parameters
    ----------
    df : pandas.DataFrame
        Data for which clusters and centers will be plotted.
    ncluster : INT
        Number of clusters.
    y1 : INT
        Column 1
    y2 : INT
        Column 2

    Returns
    -------
    df : pandas.DataFrame
        Data with cluster labels column added.
    centres : array
        Cluster Centers.

    '''
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df)

    lab = kmeans.labels_
    df['labels'] = lab
    # extract the estimated cluster centres
    centres = kmeans.cluster_centers_

    centres = np.array(centres)
    xcen = centres[:, 0]
    ycen = centres[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    sc = plt.scatter(df[y1], df[y2], 10, lab, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(f"Energy Use per capita({y1})")
    plt.ylabel(f"Energy Use per capita({y2})")
    plt.legend(*sc.legend_elements(), title='clusters')
    plt.title('Clusters of Energy Use per capita in 1990 and 2010')
    plt.show()

    print()
    print(centres)

    return df, centres


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 
    and growth rate g"""

    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


def forecast_energy(data, country, start_year, end_year):
    '''
    forecast_energy will analyse data and optimize to forecast Energy Use in 
    Kg Oil equivalent per capita of selected country

    Parameters
    ----------
    data : pandas.DataFrame
        Data for which forecasting analysis is performed.
    country : STR
        Country for which forecasting is performed.
    start_year : INT
        Starting year for forecasting.
    end_year : INT
        Ending year for forecasting.

    Returns
    -------
    None.

    '''
    data = data.loc[:, country]
    data = data.dropna(axis=0)

    energy = pd.DataFrame()

    energy['Year'] = pd.DataFrame(data.index)
    energy['Energy'] = pd.DataFrame(data.values)
    energy["Year"] = pd.to_numeric(energy["Year"])
    importlib.reload(opt)

    param, covar = opt.curve_fit(logistic, energy["Year"], energy["Energy"],
                                 p0=(1.2e12, 0.03, 1990.0))

    sigma = np.sqrt(np.diag(covar))

    year = np.arange(start_year, end_year)
    forecast = logistic(year, *param)
    low, up = err.err_ranges(year, logistic, param, sigma)
    plt.figure()
    plt.plot(energy["Year"], energy["Energy"], label="Energy Use")
    plt.plot(year, forecast, label="Forecast", color='k')
    plt.fill_between(year, low, up, color="yellow", alpha=0.7,
                     label='Confidence Margin')
    plt.xlabel("Year")
    plt.ylabel("Energy Use in Kg Oil equivalent per capita")
    plt.legend()
    plt.title(f'Energy Use in Kg Oil equivalent forecast for {country}')
    plt.savefig(f'{country}.png', bbox_inches='tight', dpi=300)
    plt.show()

    energy2030 = logistic(2030, *param)/1e9

    low, up = err.err_ranges(2030, logistic, param, sigma)
    sig = np.abs(up-low)/(2.0 * 1e9)
    print()
    print(f"Energy Use in Kg Oil by 2030 in {country}",
          np.round(energy2030*1e9, 2), "+/-", np.round(sig*1e9, 2))


#Reading Energy Use in Kg oil equivalent per capita Data
energy = reading_data("energy_use_kg_oil.csv")
print(energy.describe())

#Finding transpose of Energy Use in Kg oil equivalent per capita Data
energy_tr = transpose(energy)
print(energy_tr.head())

#Selecting years for which correlation is done for further analysis
energy = energy[["1990", '1995', "2000", '2005', "2010", '2015']]
print(energy.describe())

correlation_and_scattermatrix(energy)
column1 = "1990"
column2 = "2010"

# Extracting columns for clustering
energy_ex = energy[[column1, column2]]
energy_ex = energy_ex.dropna(axis=0)

# Normalising data and storing minimum and maximum
energy_norm, df_min, df_max = ct.scaler(energy_ex)

print()
print("Number of Clusters and Scores")
ncluster = cluster_number(energy_ex, energy_norm)

energy_norm, cen = clusters_and_centers(energy_norm, ncluster, column1,
                                        column2)

#Applying backscaling to get actual cluster centers
scen = ct.backscale(cen, df_min, df_max)
print('scen\n', scen)

energy_ex, scen = clusters_and_centers(energy_ex, ncluster, column1, column2)


print()
print('Countries in cluster 1')
print(energy_ex[energy_ex['labels'] == 1].index.values)


#Forecast Energy Use per capita for Canada
forecast_energy(energy_tr, 'Canada', 1960, 2031)

#Forecast Energy Use per capita for United States
forecast_energy(energy_tr, 'United States', 1960, 2031)

#Forecast Energy Use per capita for Finland
forecast_energy(energy_tr, 'Finland', 1960, 2031)

#Forecast Energy Use per capita for Japan
forecast_energy(energy_tr, 'Japan', 1960, 2031)
