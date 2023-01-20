# 1.Data Pre-procesing Step
# importing libraries
import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# Extracting Independent and dependent Variable
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#independent variable feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2.Fitting Logistic Regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# 3.Predicting the test set result
y_pred = classifier.predict(x_test)

# 4.Creating the Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 5.Visualizing the test set result
x_set, y_set = x_train, y_train

x1, x2 = nm.meshgrid(nm.arange(start =x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
nm.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))
plt.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(x1.min(), x1.max())  
plt.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c=ListedColormap(('purple', 'green'))(i), label=j)
plt.title('Logistic Regression (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')  
plt.legend()  
plt.show()  
