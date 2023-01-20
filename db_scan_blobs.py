import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


noise = list()
cluster0 = list()
cluster1 = list()
cluster2 = list()

for i in range(len(labels)):
    if labels[i] == 0:
        cluster0.append(labels_true[i])  # I use labels_true[i] to access the content of labels DataFrame which \
        # contains the "target" blobs
    elif labels[i] == 1:
        cluster1.append(labels_true[i])
    elif labels[i] == 2:
        cluster2.append(labels_true[i])
    else:
        noise.append(labels_true[i])

noiseless_clusters = list(labels)
while noiseless_clusters.count(-1) > 0:
    noiseless_clusters.remove(-1)

noiseless_X = []
noise_of_X = []
for x_axis in range(len(X)):
    if labels[x_axis] != -1:
        noiseless_X.append(X[x_axis])
    else:
        noise_of_X.append(X[x_axis])
noiseless_X = np.array(noiseless_X)
noise_of_X = np.array(noise_of_X)

# Logistic Regression #

logistic = LogisticRegression(random_state=0)

# KANΩ LOGISTIC REGRESSION μεταξύ θορύβου και των core_samples που είναι "αθόρυβα"   #

logistic.fit(noiseless_X, noiseless_clusters)
# logistic.fit(noiseless_X, y_pred)

# ==== ΚΑΝΩ y predict το θόρυβο πάνω στη Logistic =======================================#

y_pred = logistic.predict(noise_of_X)

# =================  Classification Report ===========================================================#
y_true = noise
target_names = ['blobs: 0', 'blobs : 1', 'blobs: 2']
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.scatter(noiseless_X[:, 0], noiseless_X[:, 1], c=noiseless_clusters)
plt.show()

