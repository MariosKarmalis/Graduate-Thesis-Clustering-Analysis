import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

sys.setrecursionlimit(10**5)


def k_mean(file, clusters, target):
    """ Custom  function of the k-means clustering algorithm with logistic regression
     Giving the designated data file, no. of clusters and target column to try and cluster """

    # Reading the csv file to obtain the needed data
    data = pd.read_csv(file)

# ======== START: One-hot Encode the 'Ineligible Column' attributes/ Manual Pre-processing ============== #

    # START- Cleaning and modifying the data : for Wisconcin Breast Cancer.csv
    data = data.drop('id', axis=1)
    data = data.drop('Unnamed: 32', axis=1)
    # Mapping Benign to 0 and Malignant to 1
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    # Scaling the dataset
    datas = pd.DataFrame(preprocessing.scale(data.iloc[:, 1:32]))
    datas.columns = list(data.iloc[:, 1:32].columns)
    datas['diagnosis'] = data['diagnosis']
    # END- Cleaning and modifying the data : for Wisconcin Breast Cancer.csv


    '''
    # START- Cleaning and modifying the data : for Churn Modeling.csv
    label_encoder = preprocessing.LabelEncoder()
    data.drop(['RowNumber', 'Surname', 'CustomerId'], 1, inplace=True)
    data.Geography = label_encoder.fit_transform(data.Geography)

    # similarly doing label encoding for other categorical columns
    data.Gender = label_encoder.fit_transform(data.Gender)
    # one_hot = pd.get_dummies(data['Unwanted Column'])
    # END- Cleaning and modifying the data : for Churn Modeling.csv
    '''
    # Delete the 'target' attribute from the dataset

    y = data[target]
    data = data.drop(columns=target, axis=1)
    # print(data.head())

    #  ===  Handling of the missing values, if any. === #
    # data.fillna(method='ffill', inplace=True)

# ======== END: One-hot Encode the 'Ineligible Column' attributes/ Manual Pre-processing ============== #

    # Scaling the data to bring all the attributes to a comparable level
    X = StandardScaler().fit_transform(data)
    normalized_X = normalize(X)

    # Creating k-means clustering instance and fitting the scaled data 'X'
    km = KMeans(init="random", n_clusters=clusters, n_init=10)
    km.fit(normalized_X)
    y_pred = km.labels_
    M , B = list(), list()
    for i in y_pred:
        if i == 1:
            M.append(i)
        else:
            B.append(i)

    print("K-MEANS Results: \n\n Malignant Cases: %d \n Benign Cases: %d" %(M.count(1),B.count(0)))
    plt.scatter(normalized_X[:, 0], normalized_X[:, 1], c=km.labels_)
    plt.title("K-Means Clustering Algorithm: \n \
    Number of Clusters: {}".format(len(set(y_pred))))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(normalized_X[:, 0], y_pred)))
    print('Completeness: {}'.format(metrics.completeness_score(normalized_X[:, 0], y_pred)))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(normalized_X, y_pred))
    plt.show()


def db_scan(file, epsilon, target, clusters, min_samps):
    """ Custom DBSCAN function with user designated data file and target column data to drop during pre-processing
        , epsilon density margin ,wanted  no. of clusters
        AND minimum no. of samples to use in dbscan instance
    """

    # Reading the csv file to obtain the needed data
    data = pd.read_csv(file)


# ======== START: One-hot Encode the 'Ineligible Column' attributes/ Manual Pre-processing ============== #

    # START - Cleaning and modifying the data : for Wisconcin Breast Cancer.csv - START
    data = data.drop('id', axis=1)
    data = data.drop('Unnamed: 32', axis=1)
    # Mapping Benign to 0 and Malignant to 1
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    # Scaling the dataset
    datas = pd.DataFrame(preprocessing.scale(data.iloc[:, 1:32]))
    datas.columns = list(data.iloc[:, 1:32].columns)
    datas['diagnosis'] = data['diagnosis']
    # END - Cleaning and modifying the data : for Wisconcin Breast Cancer.csv - END


    '''
    # START- Cleaning and modifying the data : for Churn Modeling.csv -START
    label_encoder = preprocessing.LabelEncoder()
    data.drop(['RowNumber', 'Surname', 'CustomerId'], 1, inplace=True)
    data.Geography = label_encoder.fit_transform(data.Geography)

    # similarly doing label encoding for other categorical columns
    data.Gender = label_encoder.fit_transform(data.Gender)

    # END- Cleaning and modifying the data : for Churn Modeling.csv -END
    '''
    # Delete the 'target' attribute from the dataset

    y = data[target]

    data = data.drop(columns=target, axis=1)
    #  ===  Handling of the missing values, if any. === #
    # data.fillna(method='ffill', inplace=True)

# ======== END: One-hot Encode the 'Ineligible Column' attributes/ Manual Pre-processing ============== #

    # Scaling the data to bring all the attributes to a comparable level
    X = StandardScaler().fit_transform(data)
    normalized_X = normalize(X)

    # Exporting normalized input X to external txt file for inspection
    norm_file = np.savetxt('normalized.txt', normalized_X, delimiter=',')

    #  Creating a custom parameter DBSCAN instance function

    def db_scan_wrapper(e, cluster, samples):

        global eps
        flag = False
        scan = DBSCAN(eps=e, min_samples=samples)
        fitted_data = scan.fit(normalized_X)
        labels = fitted_data.labels_
    #    labels = scan.fit_predict(X)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters_ == cluster:
            flag = True
            eps = e
        while not flag:
            db_scan_wrapper(e + 0.01, cluster, samples)
            break
        return eps

    # ==================== Computing the DBSCAN algorithm ===========================================#

    eps = db_scan_wrapper(epsilon, clusters, min_samps)
    print("\nThe most appropriate epsilon was: ", eps)
    dbscan = DBSCAN(eps=eps, min_samples=min_samps).fit(normalized_X)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_
    # print(labels)

    noise = list()
    cluster0 = list()
    cluster1 = list()
    '''cluster2 = list()
    cluster3 = list()
    cluster4 = list()
    cluster5 = list() '''
    count_c0, count_c1 = 0, 0

    # TO-DO:  Implement for loop with "index in components" to try to discern normalized data between the 6 clusters

    for i in labels:
        if i == 0:
            cluster0.append(i)
            count_c0 += 1
        elif i == 1:
            cluster1.append(i)
            count_c1 += 1
        else:
            noise.append(i)

    # print("\n", count_c0, count_c1, "\n")
    data0, data1, data2, data3, data4, data5, noisy_data = list(), list(), list(), list(), list(), list(), list()

    for i in range(len(labels)):
        if labels[i] == 0:
            data0.append(y.at[i])  # I use y[i] to access the content of y DataFrame which contains the "target" Column
        elif labels[i] == 1:
            data1.append(y.at[i])
        # elif labels[i] == 2:
        #     data2.append(y.at[i])
        # elif labels[i] == 3:
        #     data3.append(y.at[i])
        # elif labels[i] == 4:
        #     data4.append(y.at[i])
        # elif labels[i] == 5:
        #     data5.append(y.at[i])
        else:
            noisy_data.append(y.at[i])

    print("Cluster labeled 1 size: %d with %d '1' contents.  \
          Cluster labeled 0 size: %d  with %d '0' contents.  \
         " % (len(data0), data0.count(1), len(data1), data1.count(0)))

#  len(data2), data2.count(),
    #               len(data3), data3.count(), len(data4), data4.count(), len(data5), data0.count()
    '''  
    Cluster labeled 2 size: %d with %d '2' contents.\n  \
    Cluster labeled 3 size: %d  with %d '3' contents.\n \
    Cluster labeled 2 size: %d with %d '4' contents.\n  \
    Cluster labeled 2 size: %d with %d '5' contents.\n   
    '''

    noiseless_clusters = list(labels)
    while noiseless_clusters.count(-1) > 0:
        noiseless_clusters.remove(-1)

    #  Getting noiseless rows of X array for further processing START
    noiseless_X = []
    noise_of_X = []
    for x_axis in range(len(normalized_X)):
        if dbscan.labels_[x_axis] != -1:
            noiseless_X.append(normalized_X[x_axis])
        else:
            noise_of_X.append(normalized_X[x_axis])
    noiseless_X = np.array(noiseless_X)
    noise_of_X = np.array(noise_of_X)

    #  Getting noiseless rows of X array for further processing  END

    print('\n Clusters data where label is : %s \n' % data0)
    print('\n Clusters data where label is : %s \n' % data1)
    print('Noise points in dataset: ', noisy_data)

    '''
    print('\n Clusters data where label is 2: %s \n' % data2)
    print('\n Clusters data where label is 3: %s \n' % data3)
    print('\n Clusters data where label is 4: %s \n' % data4)
    print('\n Clusters data where label is 5: %s \n' % data5)
    '''

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('\n Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d\n' % n_noise_)

    # Classification Report before using Logistic Regression
    # y_true = data0 + data1
    y_true = cluster0 + cluster1
    y_predict = noiseless_clusters
    target_names = ["1s: ", "0s: "]
    # target_names = ['gender: Male', 'gender: Female']
    print("Pre-logistic class report:\n", classification_report(y_true, y_predict, target_names=target_names, zero_division=0))

    # TA noiseless samples προσθέτω για train στο y_predict
    # y_pred = dbscan.fit_predict(noiseless_X, noiseless_clusters)

    # Logistic Regression #

    logistic = LogisticRegression(random_state=0)
    # KANΩ LOGISTIC REGRESSION μεταξύ θορύβου \# και των core_samples που είναι "αθόρυβα"   #
    logistic.fit(noiseless_X, noiseless_clusters)
    # logistic.fit(noiseless_X, y_pred)

    # ==== ΚΑΝΩ y predict το θόρυβο πάνω στη Logistic =======================================#
    y_pred = logistic.predict(noise_of_X)

    # =================  Classification Report - START ===========================================================#
    y_true = noisy_data
    target_names = ['1s :', '0s: ']
    # target_names = ['gender: Female', 'gender: Male']
    # target_names = ['Withdrew: 0', 'Withdrew: 1']
    print("Logistic Regression Class Report:\n", classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    # =================  Classification Report - END ===========================================================#


    # Visualize Clusters and output evaluation metrics to console #

    plt.scatter(noiseless_X[:, 0], noiseless_X[:, 1], c=noiseless_clusters, marker='o')
    # plt.scatter(noise_of_X[:, 0], noise_of_X[:, 1], c=y_pred, marker='o')
    plt.title('DBSCAN: \n Number of clusters: {}'.format(len(set(y_pred[np.where(y_pred != -1)]))))
    print('Homogeneity: {}'.format(metrics.homogeneity_score(noiseless_X[:, 0], noiseless_clusters)))
    print('Completeness: {}'.format(metrics.completeness_score(noiseless_X[:, 0], noiseless_clusters)))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(noiseless_X, noiseless_clusters))
    plt.show()

# ==========  DBSCAN: Function Calling ================================================= #

#db_scan('heart.csv', 0.937, 'target', 2, 14)
# db_scan('gender_classification_v7.csv', 0.67, 'gender', 2, 25)
#db_scan('Churn Modeling.csv', 0.22, 'Exited', 2, 10)
# db_scan('Churn Modeling.csv', 1.2, 'Exited', 2, 2500)
db_scan('wisconsin.csv', 0.67, 'diagnosis', 2, 10)

# ==========  K-MEANS : Function Calling =============================================== #

#k_mean('Churn Modeling.csv', 2, 'Exited')
k_mean('wisconsin.csv', 2, 'diagnosis')

# TODO: CROSS-CHECK class report before logistic regression
# DONE: ADD K-MEANS / OTHER ML ALGORITHM for cross referencing data results
# TODO: ADD "weird" behaviour of low samples+low epsilon clustering VS higher min samples with higher epsilon clustering

