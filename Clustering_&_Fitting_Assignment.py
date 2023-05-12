# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:18:50 2023

@author: ibrah
"""

#importing the libraries that will used for this analysis
import cluster_tools as ct
import errors as err
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

### Defining functions for the study

def data(filename):
    """
    A function that reads in GDP_per_capital data from world bank library
    and output a dataframe and its transpose

    Parameter:
        filename: the file to be read in as a dataframe

    Returns:
        GDP_per_capita dataframe and its transpose
    """

    # using pandas to read in dataset, with the first 4 rows skipped
    worldB_df = pd.read_csv(filename, skiprows=4)

    return worldB_df, worldB_df.transpose()


# Polynomial function for the fitting part of the analysis

def poly(t, *coefficients):
    """Computes a polynomial function for fitting analysis

    Parameters:
        t: The time
        coefficients: Coefficients of the polynomial

    Returns:
        The GDP_per_capital for different years
    """
    return np.polyval(coefficients, t)


def poly_fit(data, degree, predicted_years, Countries):
    """plots polynomial fit for different countries and makes 20 years predictions
       for each country

    Parameters:
        data: dataframe used for this analysis
        degree: gthe degree of polynomial fit
        predicted year: prediction for years 2031 and 2041
        Countries: countries considered for fittings

    Returns:
        A plot of GDP per capital of each country and its prediction
    """

    years = data['Year'].values
    GDP_per_capita = data[Countries].values

    # Fitting the polynomial model
    coefficients = np.polyfit(years, GDP_per_capita, degree)

    # Predictions for the given years
    predictions = []
    for year in predicted_years:
        prediction = np.polyval(coefficients, year)
        predictions.append(prediction)

    # Generating points for the fitted curve
    curve_years = np.linspace(min(years), max(years), 100)
    extended_curve_years = np.concatenate((curve_years, predicted_years))

    gdp_curve = np.polyval(coefficients, extended_curve_years)

    # Error range
    residuals = GDP_per_capita - np.polyval(coefficients, years)
    sigma = np.std(residuals)
    lower = gdp_curve - sigma
    upper = gdp_curve + sigma

    # Plotting the data, fitted curve, and error range
    plt.figure(dpi=300)
    plt.plot(years, GDP_per_capita, 'bo', label='Data')
    plt.plot(extended_curve_years, gdp_curve, 'r-', label='Fitted Curve')
    plt.fill_between(extended_curve_years, lower, upper, color='yellow', alpha=0.4, label='Error Range')
    plt.xlabel('Year')
    plt.ylabel('GDP_per_capita (current US$)')
    plt.title(f'{Countries}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{Countries}.png')
    plt.show()

    return predictions, gdp_curve, (lower, upper)



#calling the data function to read in gdp data
gdp_df, gdp_T = data('GDP_per_capital.csv')
#print(gdp_df)
#print(gdp_df.describe())

#cleaning the dataframe

#dropping unused columns from the gdp dataframe
gdp_df=gdp_df.drop(['Country Code','Indicator Name','Indicator Code'], axis=1)

#reseting the index
gdp_df.reset_index(drop=True, inplace=True)

# Extracting  years from dataset(considering years with 10 years interval)

gdp_data=gdp_df[['Country Name','1981','1991', '2001','2011','2021']]
#print(gdp_data)
#print(gdp_data.describe())

#Checking for missing values
#print(gdp_data.isna().sum())


#removing rows with missing values from the data set
gdp_data1= gdp_data.dropna()
#print(gdp_data1)
#print(gdp_data1.describe())

#set country name as index
gdp_data2 = gdp_data1.set_index('Country Name')
#print(gdp_data2)


# Checking for correlation between the chosen years

# Correlation
corr=gdp_data2.corr()
#print(corr)

# creating heatmap to show correlation plot
ct.map_corr(gdp_data2)

#plotting scatter matrix, to aid correlation analysis
pd.plotting.scatter_matrix(gdp_data2, figsize=(10,10), s=5, alpha=0.8)

#prevents labels overlap
plt.tight_layout()
#show scatter_matrix
plt.show()

"""The correlation between the data for the years 1981 and 2021 appears to be
 relatively low, based on both the correlation map and scatter matrix.
 As a result, these two years will be used for the clustering analysis."""


# extracting  the two columns for clustering analysis
gdp_cluster=gdp_data2[['1981', '2021']].copy()
gdp_cluster

# Normalizing the data
gdp_norm, gdp_min, gdp_max = ct.scaler(gdp_cluster)
#print(gdp_norm)
#print(gdp_min)
#print(gdp_max)

print()
print("no_of_cl   silhouette score")

for x in range(2, 10):
    # setting up the no of expected clusters(kmeans) and fitting data into kmeans object
    kmeans = cluster.KMeans(n_clusters=x)
    kmeans.fit(gdp_cluster)

    # extract labels and calculate silhoutte scores
    labels = kmeans.labels_

   # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    #calculating the silhouette score for the clusters
    print (x,skmet.silhouette_score(gdp_cluster, labels))

"""The silhouette score suggests that clusters 2 and 3 are both viable options.
 These clusters will be further tested to determine the best one to use."""


nclusters = 2 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nclusters)
kmeans.fit(gdp_norm)

# extract labels
labels = kmeans.labels_

# extracting the estimated number of cluster centers
cen = kmeans.cluster_centers_

cm = plt.cm.get_cmap('tab10')

plt.figure(figsize=(6.0, 6.0))

# scatter plot when number of cluster is 2
plt.scatter(gdp_norm["1981"], gdp_norm["2021"], c=labels, cmap=cm, label='Clusters')

# show cluster centres
xcen = cen[:,0]
ycen = cen[:,1]
plt.scatter(xcen, ycen, color="k", marker="d", s=80, label='Centroid')

plt.xlabel("GDP(1981)")
plt.ylabel("GDP(2021)")
plt.title("Two- Cluster Plot of GDP Per Capita for 1981 and 2021")
plt.legend()
plt.show()


nclusters = 3 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nclusters)
kmeans.fit(gdp_norm)

# extract labels
labels = kmeans.labels_

# extracting the estimated number of cluster centers
cen = kmeans.cluster_centers_

cm = plt.cm.get_cmap('tab10')

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(gdp_norm["1981"], gdp_norm["2021"], c=labels, cmap=cm, label='Clusters')

# show cluster centres
xcen = cen[:,0]
ycen = cen[:,1]
plt.scatter(xcen, ycen, color="k", marker="d", s=80, label='Centroid')

plt.xlabel("GDP(1981)")
plt.ylabel("GDP(2021)")
plt.title("Three- Cluster Plot of GDP Per Capita for 1981 and 2021")
plt.show()

# The plot was more meaningful when the number of clusters was set to 3.

#showing the different countries in each cluster
# add the cluster labels as a new column
gdp_data2['Cluster'] = labels

# group the data by cluster label and print the countries in each cluster
for label, group in gdp_data2.groupby('Cluster'):
    print(f"Countries in Cluster {label}:\n {', '.join(group.index.tolist())}")


# scaling  the clustered data back to the original scale and create a plot using the original data.

plt.style.use('seaborn')
plt.figure(dpi=300)

# now using the original dataframe
scatter = plt.scatter(gdp_data2["1981"], gdp_data2["2021"], c=labels, cmap="tab10", label='Cluster')

# rescale and show cluster centres
scen = ct.backscale(cen, gdp_min, gdp_max)
xcen = scen[:,0]
ycen = scen[:,1]

#creating a scattered plot
centroid = plt.scatter(xcen, ycen, c="k", marker="d", s=80,  label='Centroid')

plt.xlabel("1981")
plt.ylabel("2021")
plt.title("GDP per capita (current US$)")
plt.legend()
plt.savefig("GDP_PER_CAPITAL.PNG")
plt.show()


# Now, we will proceed to the fitting part of the assignment.
# the gdp_per_capital data will be used for fitting and prediction
#two countries each  were selected from the 2 of the data clusters
#Portugal  and South Africa was selected from cluster 0
#China and United States was selected from cluster 2

#calling the transposed gdp_data for fitting analysis
print(gdp_T)

#cleaning the transposed data
gdp_T.columns=gdp_T.iloc[0]
gdp_T=gdp_T.iloc[1:]

gdp_T = gdp_T.drop(['Country Code', 'Indicator Code', 'Indicator Name',
                    'Unnamed: 66'], axis=0)

gdp_T.reset_index(inplace=True)
gdp_T.rename(columns={'index':'Year'}, inplace=True)
gdp_T.rename(columns={'Country Name': 'Year'}, inplace=True)
gdp_T=gdp_T.apply(pd.to_numeric)
#print(gdp_T.dtypes)

gdp_all= gdp_T[gdp_T['Year'].isin([1981,1991,2001,2011,2021])]
print(gdp_all)

#generating dataframes from gdp_ all for the countries we are considering
gdp_Portugal= gdp_all['Portugal']
gdp_SA= gdp_all['South Africa']
gdp_China= gdp_all['Monaco']
gdp_US= gdp_all['United States']


#calling the functions, poly(t, *coefficients) and def poly_fit(data, degree, predicted_years, Countries)
#to fit and make  plots with predictions


##########   Portugal fitting and Prediction  ##########

predicted_years = np.array([2031, 2041])
predictions, gdp_curve, error_range = poly_fit(gdp_all, 3, predicted_years, 'Portugal')

print("prediction for 2031:", predictions[0])
print("prediction for 2041:", predictions[1])


##########   South Africa fitting and Prediction  ##########

predicted_years = np.array([2031, 2041])
predictions, gdp_curve, error_range = poly_fit(gdp_all, 3, predicted_years, 'Burundi')

print("prediction for 2031:", predictions[0])
print("prediction for 2041:", predictions[1])


##########   China fitting and Prediction  ##########

predicted_years = np.array([2031, 2041])
predictions, gdp_curve, error_range = poly_fit(gdp_all, 3, predicted_years, 'Monaco')

print("prediction for 2031:", predictions[0])
print("prediction for 2041:", predictions[1])


##########   United States fitting and Prediction  ##########

predicted_years = np.array([2031, 2041])
predictions, gdp_curve, error_range = poly_fit(gdp_all, 3, predicted_years, 'United States')

print("prediction for 2031:", predictions[0])
print("prediction for 2041:", predictions[1])

































