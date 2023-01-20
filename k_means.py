import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler


'''
# generate synthetic two-dimensional data
X, y = make_blobs(random_state=1)
# build the K-means clustering  model
k_means = KMeans(init="random", n_clusters=3)
k_means.fit(X)
labels = k_means.labels_
ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='jet', marker='o')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], c=[0, 1, 2], cmap='jet',
            marker='^', edgecolors='k')

plt.show()

'''
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
# rescale the data to zero mean and unit variance
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
# plot the cluster assignments
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='jet', s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()