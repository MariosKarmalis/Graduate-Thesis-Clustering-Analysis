import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression


sys.setrecursionlimit(10**5)
# Reading the csv file to obtain the needed data

data = pd.read_csv("6 class csv.csv")

# One-hot Encode the 'Spectral Class' attribute
one_hot = pd.get_dummies(data['Spectral Class'])

# Delete the 'target' attribute from the dataset

y = data['Star type']
# print(y.at[80])
data = data.drop(['Star color', 'Star type', 'Spectral Class'], axis=1)
data = data.join(one_hot)


# Handling the missing values
# data.fillna(method='ffill', inplace=True)

# Scaling the data to bring all the attributes to a comparable level
X = StandardScaler().fit_transform(data)
normalized_X = normalize(X)

# Exporting normalized input X to external txt file for inspection
norm_file = np.savetxt('normalized.txt', normalized_X, delimiter=' ')


#  Creating a custom parameter DBSCAN instance
#  epsilon = 0.778 OPTIMAL FOR 2 CLUSTERS with 10 min_samples


def db_scan_wrapper(e, cluster):
    global eps
    flag = False
    scan = DBSCAN(eps=e, min_samples=3)
    fitted_data = scan.fit(normalized_X)
    labels = fitted_data.labels_
#    labels = scan.fit_predict(normalized_X)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters_ == cluster:
        flag = True
        eps = e
    while not flag:
        db_scan_wrapper(e + 0.001, cluster)
        break
    return eps

#  Computing the DBSCAN algorithm


eps = db_scan_wrapper(0.37, 6)
# eps = db_scan_wrapper(0.301, 6)
print(eps)
scan = DBSCAN(eps=eps, min_samples=3).fit(normalized_X)
# db_scan = DBSCAN(eps=0.45, min_samples=10).fit(normalized_X)
core_samples_mask = np.zeros_like(scan.labels_, dtype=bool)
core_samples_mask[scan.core_sample_indices_] = True
labels = scan.labels_
# print(labels)

'''
noise = list()
cluster0 = list()
cluster1 = list()
cluster2 = list()
cluster3 = list()
cluster4 = list()
cluster5 = list()
count_c0, count_c1, count_c2, count_c3, count_c4, count_c5 = 0, 0, 0, 0, 0, 0


# TO-DO:  Implement for loop with "index in components" to try to discern normalized data between the 6 clusters

for i in labels:
    if i == 0:
        cluster0.append(i)
        count_c0 += 1
    elif i == 1:
        cluster1.append(i)
        count_c1 += 1
    elif i == 2:
        cluster2.append(i)
        count_c2 += 1
    elif i == 3:
        cluster3.append(i)
        count_c3 += 1
    elif i == 4:
        cluster4.append(i)
        count_c4 += 1
    elif i == 5:
        cluster5.append(i)
        count_c5 += 1
    else:
        noise.append(i)
'''

data0, data1, data2, data3, data4, data5, noisy_data = list(), list(), list(), list(), list(), list(), list()
for i in range(len(labels)):
    if labels[i] == 0:
        data0.append(y.at[i])  # Use y.at[i] to access the content of y DataFrame which contains the "Star Type" Column
    elif labels[i] == 1:
        data1.append(y.at[i])
    elif labels[i] == 2:
        data2.append(y.at[i])
    elif labels[i] == 3:
        data3.append(y.at[i])
    elif labels[i] == 4:
        data4.append(y.at[i])
    elif labels[i] == 5:
        data5.append(y.at[i])
    else:
        noisy_data.append(y.at[i])


noiseless_clusters = list(labels)
while noiseless_clusters.count(-1) > 0:
    noiseless_clusters.remove(-1)

print(noiseless_clusters.count(0), noiseless_clusters.count(1), noiseless_clusters.count(2),
      noiseless_clusters.count(3), noiseless_clusters.count(4), noiseless_clusters.count(5))
# Getting noiseless rows of normalized_X array for further processing START
noiseless_X = []
noise_of_X = []
for x_axis in range(len(normalized_X)):
    if scan.labels_[x_axis] != -1:
        noiseless_X.append(normalized_X[x_axis])
    else:
        noise_of_X.append(normalized_X[x_axis])
noiseless_X = np.array(noiseless_X)
noise_of_X = np.array(noise_of_X)


# Getting noiseless rows of normalized_X array for further processing  END

print('\n Clusters data where label is 0: %s \n' % data0)
print('\n Clusters data where label is 1: %s \n' % data1)
print('\n Clusters data where label is 2: %s \n' % data2)
print('\n Clusters data where label is 3: %s \n' % data3)
print('\n Clusters data where label is 4: %s \n' % data4)
print('\n Clusters data where label is 5: %s \n' % data5)
print('Noise points in dataset: ', noisy_data)


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('\n Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# TO-DO ?? Predict y for noiseless data and noiseless labels


y_pred = scan.fit_predict(noiseless_X, noiseless_clusters)  # TA noiseless samples προσθέτω για train στο y_predict
# print("\n y_pred contents: \n", y_pred, "\n With length: ", len(y_pred), "\n")


# Logistic Regression #

logistic = LogisticRegression(random_state=0)

# KANΩ LOGISTIC REGRESSION μεταξύ θορύβου και των core_samples που είναι "αθόρυβα"   #

# logistic.fit(noiseless_X, noiseless_clusters)
logistic.fit(noiseless_X, y_pred)

#  ΚΑΝΩ y predict το θόρυβο πάνω στη Logistic =======================================#

# y_pred = logistic.predict(noise_of_X)


# Visualize Clusters and output evaluation metrics to console #

plt.scatter(noiseless_X[:, 0], noiseless_X[:, 1], c=y_pred, marker='o')
plt.title('Number of clusters: {}'.format(len(set(y_pred[np.where(y_pred != -1)]))))


# print('Homogeneity: {}'.format(metrics.homogeneity_score(noiseless_X[:, 0], y_pred)))
# print('Completeness: {}'.format(metrics.completeness_score(noiseless_X[:, 0], y_pred)))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(noiseless_X, y_pred))

# score = logistic.score(noise_of_X, y_pred)
# score = logistic.score(noise, y_pred)
# print('Mean accuracy: %0.3f' % score)
plt.show()

# ΤΟ-DO !!! -->  NA ANTIΣΤΟΙΧΗΣΩ ΤΑ STAR COLOR TΗΣ ΜΕΤΑΒΛΗΤΗΣ y Mε τα clusters του Normalized_Χ
