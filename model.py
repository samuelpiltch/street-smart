import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Implementation of k-means cluster algorithm with descriptions of model steps and procedure.
# Useful algorithm for clustering of unclassified data.

original_data = pd.read_csv('./data/street-smart.csv')
data = original_data.dropna()
data = np.matrix(data.drop(["Latitude", "Longitude", "ZIPCode", "CensusTract"], 1), dtype=np.int64)

# The amount of clusters or classificiations you want to output.
desired_clusters = 5

# Cluster Centroids are the center of your clusters. K-means algorithm starts by picking random centroids.
# Commonly picks two random example points and sets them as your centroids.
centroids = {}
classification_clusters = {}
cluster_labels = []

def setRandomClusterCentroid():
    for cluster in range(desired_clusters):
        centroids[cluster] = data[cluster]

# For each training example, find the nearest cluster centroid and assign that point to that cluster.
def getNearPoints():
    # Make each cluster start with an empty array where training examples will be stored
    for cluster in range(desired_clusters):
        classification_clusters[cluster] = []

    # For every data point, calculate it's distance from each centroid (using NumPy norm.)
    # Then find the index of the closer centroid. Append the data point to the cluster.
    for data_point in data:
        distances = [np.linalg.norm(data_point-centroids[centroid]) for centroid in centroids]
        classification = distances.index(min(distances))
        classification_clusters[classification].append(data_point)
        cluster_labels.append(classification)

# Finds average value of points for each cluster. Then moves cluster centroid to average of cluster,
# which will get the cluster centroid closer to the desired central cluster position.
def moveClusterCentroid():
    for cluster in classification_clusters:
        centroids[cluster] = np.average(classification_clusters[cluster], axis=0)

def predict(data):
    distances = [np.linalg.norm(data - centroids[centroid]) for centroid in centroids]
    classification = distances.index(min(distances))

setRandomClusterCentroid()

max_iter = 100000
for iteration in range(max_iter):
    classification_clusters = {}
    cluster_labels = []
    getNearPoints()
    moveClusterCentroid()
    print "Iteration: ", iteration + 1

original_data = original_data.dropna()
original_data = original_data.assign(Classification = cluster_labels)

date = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
original_data.to_csv('export/' + date + '.csv', index = False)
print "Exported to `export/" + date + ".csv`"
