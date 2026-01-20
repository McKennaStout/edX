# Module 4 — Key concepts

## Clustering

- Definition (plain): Clustering groups similar things together without labels.
- Definition (formal): Clustering groups data points into clusters so points in the same cluster are more similar than points in different clusters.
- Support (professor phrasing):
  - In analytics, Clustering means taking a set of data points and dividing them up into groups so each group contains points that are close to each other or similar.
  - Here's a graph of some points and a natural way to group them into three clusters so that for each cluster, the points in each cluster are close to each other.
  - Let's see some examples of when clustering might be useful.

## k-Means

- Definition (plain): k-means picks k centers and assigns points to the closest center, repeating until it settles.
- Definition (formal): k-means partitions data into k clusters by minimizing within-cluster sum of squared distances to cluster centers.
- Support (professor phrasing):
  - To see how the k-means algorithm works, let's look at an example of clustering.
  - As you can see in the picture, the clusters the k-means algorithm found are pretty much what we might've picked out by hand.
  - when you have a lot of attributes, that's a lot of dimensions and we need something like k-means to find a good clustering.

## Cluster centroid

- Definition (plain): The centroid is the ‘average point’ of a cluster.
- Definition (formal): A centroid is the mean of points assigned to a cluster (the cluster center in k-means).
- Support (professor phrasing):
  - let's use z_kj to denote the jth dimension coordinate of cluster center k.
  - What we'd like to find is a set of k cluster centers and assignments of each data point to a cluster center to minimize the total distance from each data point to its cluster center.
  - we temporarily assign each data point to the cluster center it's closest to.

## Within-cluster sum of squares (WCSS)

- Definition (plain): WCSS measures how spread out points are inside clusters.
- Definition (formal): WCSS measures cluster tightness by summing squared distances of points to their assigned centroids.

