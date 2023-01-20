# Master-Thesis-Clustering-Analysis
Classification Techniques to eradicate noise in Clustering Algorithms

In this graduate thesis, we delve into the issue of noise presence in clustering 
algorithms. Specifically, our main goal is to partially reduce or even fully eradicate noise in 
known clustering algorithms via efficient practice of data classification in datasets of varying 
size. It must be stated that algorithms that function as above belong to the wider category of 
unsupervised learning algorithms. Apropos, we employ the DBSCAN clustering algorithm to 
aid us in the visualization of noise in clusters due to its innate ability to handle noise with great 
success. Furthermore, the number of clusters, while hinted at from the dataset target at hand, is 
not predetermined but becomes apparent after the clustering algorithm has been implemented 
to the dataset.
To corroborate our results, we proceeded to use the k-means clustering algorithm as a 
means of cross-validation and contrast, as opposed to the results provided by DBSCAN. At its 
core, k-means is a clustering algorithm that employs partitional clustering to split any given 
dataset into k clusters. Nevertheless, the number of clusters is designated by the user beforehand 
and is represented by the number k. Moreover, cross-checking and validation of the results is 
synonymous to precise and trustworthy conclusions, thus we implemented a set of precision 
and similarity metrics to the clustered data. Our data consists of two categories: artificially 
generated data and real-world data. As for the former (artificially generated data), we applied 
metrics such as the homogeneity metric, the completeness of the clusters and finally a 
combination of the two, the V-measure metric. Subsequently, having validated the artificial 
data provided to the DBSCAN algorithm, experimenting with the latter followed (real-world 
data). Therefore, we concluded that the classification that would provide the best results was 
the Logistic Regression classification method. To end with, we performed slight modifications 
to the default DBSCAN and k-means algorithms structure and integrated them into functions 
with customization and scalability in mind; hence being able to implement clustering to large 
size datasets applications (Big Data, Stream Computing).
