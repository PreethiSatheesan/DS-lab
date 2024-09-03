#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn import datasets ,preprocessing,neighbors
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
iris=load_iris()
print("iris data")
print(iris)
print("\n")
print("Iris feature _names")
print("\n")
print(iris.feature_names)
print("\n")
print("Integer representing features (0=setosa,1=versicolor,2=virginica)")
print("\n")
print(iris.target)
print("\n")
print(iris.target)
print("\n")
print("3 claases of target")
print("\n")
print(iris.target_names)
print("\n")
print("total of 150 pbservations and the features")
print("\n")
print(iris.data.shape)
print("\n")
x,y=iris.data[:,:],iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,random_state=0,train_size=0.7)
print("shape of train and test objects")
print("\n")
print(x_train.shape)
print(x_test.shape)
print("\n")
print(y_train)
print("\n")
print(y_test)
print("\n")
print(y_train.shape)
print(y_test.shape)
print("\n")
scalar=preprocessing.StandardScaler().fit(x_train)
x_train=scalar.transform(x_train)
print("display scaled data")
print("\n")
x_test=scalar.transform(x_test)
scores=[]
k_range=range(1,15)
for k in k_range:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
print("prediction of species:{}".format(y_pred))
print("accuracy score")
print(accuracy_score(y_test,y_pred))
print("confusion matrix")
print(confusion_matrix(y_test,y_pred))
print("Classification report")
print(classification_report(y_test,y_pred))


# In[ ]:





# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()

# Display information about the dataset
print("Iris data")
print(iris.data)
print("\nIris feature names")
print(iris.feature_names)
print("\nInteger representing features (0=setosa, 1=versicolor, 2=virginica)")
print(iris.target)
print("\n3 classes of target")
print(iris.target_names)
print("\nTotal of 150 observations and the features")
print(iris.data.shape)

# Split the dataset into training and test sets
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0, train_size=0.7)

print("\nShape of train and test objects")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")

# Standardize the features
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print("\nDisplay scaled data")
print(x_train[:5])  # Displaying first 5 rows of scaled training data

# Train and evaluate k-NN classifier with different values of k
k_range = range(1, 15)
best_k = 0
best_score = 0
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k={k}: {accuracy}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_k = k

print(f"\nBest k: {best_k} with accuracy: {best_score}")

# Display confusion matrix and classification report for the best k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(x_train, y_train)
y_pred_best = knn_best.predict(x_test)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification Report")
print(classification_report(y_test, y_pred_best))

# Ensure inline plotting if using Jupyter Notebooks
# Comment this line if running in a different environment
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:




