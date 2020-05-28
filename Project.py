#1 Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

#2 Importing the mall dataset
#dataset= pd.read_csv(r"C:\Users\Abdul Quddus\Desktop\DS-Project\games.csv")
dataset= pd.read_csv(r"X:\Git Hub\DataScience\games.csv")
#Select the annual income and the spending score columns X=dataset.iloc[:, [3,4]].values
X = dataset.values[:, 0:61]
print("\nThese are the values of the dataset:\n")
print(X)

A = dataset.iloc[:,:]
#Visualise data points
plt.scatter(A["gameId"],A["creationTime"],c='blue')
plt.title('Clusters of Game')
plt.xlabel('GameId')
plt.ylabel('CreationTime')
plt.show()

# Step 1 and 2 - Choose the number of clusters (k) and select random centroid for each cluster

#number of clusters
K=3

# Select random observation as centroids
Centroids = (A.sample(n=K))
plt.scatter(A["gameId"],A["creationTime"],c='blue')
plt.scatter(Centroids["gameId"],Centroids["creationTime"],c='red')
plt.title("Clusters of Game with random selected Centroids")
plt.xlabel('Game Id')
plt.ylabel('Creation Time')
plt.show()

Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(X)
    kmeans.fit(X)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

print("\nThese are the values of the Error that has been appended:\n")
print(Error)

kmeans = KMeans(n_clusters=3, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
# We are going to use the fit predict method that returns for each #observation which cluster it belongs to. The cluster to which #client belongs and it will return this cluster numbers into a #single vector that is  called y K-means
y_kmeans = kmeans.fit_predict(X)

print("\nThese are the values of the clusters which as assigned:\n")
print(y_kmeans)


kmeans.cluster_centers_

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
#plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
#plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, marker='*', c='yellow', label = 'Centroids')
plt.title('Clusters of Game')
plt.xlabel('Game Id')
plt.ylabel('Creation Time')
plt.legend()
plt.show()