#!/usr/bin/env python
# coding: utf-8

# In[3]:


#I choose the Iris Dataset for the analysing,the first part is CLUSTERING ANALYSIS.
#I'm going to analyse it in 3 different ways:K-means,DBSCAN and SpectralClustering.


# In[7]:


#Basic operations-generate dataset
#Import Library
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import datasets 

#Import and generate basic data set
iris = datasets.load_iris() 
#X = iris.data[:, :4]  
#means that we take four dimensions of the characteristic space:
#sepal length,sepal width,petal length and petal width

#We can see the initial format of the data we import
print("Iris data:")
print(iris)


# In[46]:


#Then wo gonna performe dimension reduction of data
#PCA dimension reduction
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets 

iris = datasets.load_iris()
x=iris.data#Use X to represent the attribute data of the dataset

#Load PCA algorithm and set the dimension after dimension reduction to 2
pca=PCA(n_components=2)

#The dimension of the original data is reduced and saved in X
X=pca.fit_transform(x)

_x,_y=[],[]
for i in range(len(reduced_x)):
    _x.append(X[i][0])
    _y.append(X[i][1])

#Data visualization after dimension reduction
plt.scatter(_x,_y,c='b',marker='o')

plt.show()


# In[45]:


#K-MEANS
from sklearn.cluster import KMeans

X=pca.fit_transform(x)
Y = iris.data[:, :4]

#Building a K-MEANS cluster
estimator = KMeans(n_clusters=3)  #Structural Cluster,k=3
estimator.fit(X)  # Clustering
Klabel = estimator.labels_  # Get the cluster tag

# Draw the K-MEANS result(PCA)
x0 = X[Klabel == 0]
x1 = X[Klabel == 1]
x2 = X[Klabel == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='group1')  
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='o', label='group2')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='o', label='group3')  
plt.legend(loc=2)  
plt.show() 

#Draw the K-MEANS result()

#As we learn from the code,the algorithme K-means demands the users to define the numbers k od the groups
#It Randomly generates K clusters, and then determine the cluster center, or directly generate K centers. 
#The clustering center of each point is determined. 
#Then calculate the new clustering center.
#Repeat the above steps until the convergence requirements are met. (usually the fixed center point does not change.

#So from its definition, we can know that when the result cluster is dense, it works better 
#Special values have great influence on the algorithme


# In[44]:


#DBSCAN
from  sklearn.cluster import DBSCAN
X=pca.fit_transform(x)
dbscan = DBSCAN(eps=0.4, min_samples=9)
#ε-neighborhood area=0.4,The minimum number of points required to form a high-density area is 9
dbscan.fit(X) 
Dlabel = dbscan.labels_
 
# Draw the DBSCAN result
x0 = X[Dlabel == 0]
x1 = X[Dlabel == 1]
x2 = X[Dlabel == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='group1')  
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='o', label='group2')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='o', label='group3')    
plt.legend(loc=2)  
plt.show()
#It starts with an arbitrary point that is not visited, and then explores the ε - neighborhood of this point. 
#If there are enough points in the ε - neighborhood, a new cluster is established, otherwise the point is labeled as noise. 
#Note that this point may be found in the ε - neighborhood of other points, and the ε - neighborhood may have enough points
#Then this point will be added to the cluster


# In[42]:


#SpectralClustering
X=pca.fit_transform(x)
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=8,eigen_solver=None,random_state=None,n_init=10,gamma=1.0,
                        affinity='rbf',n_neighbors=10,eigen_tol=0.0,assign_labels='kmeans',degree=3,coef0=1,kernel_params=None,n_jobs=1)
#n_clusters：Number of clusters；
#eigen_solver：The eigenvalue decomposition strategy is used；
#gamma：When computing a similar matrix, if you use a kernel function, the coefficients of that kernel function；
#affinity：The distance formula used to calculate similarity；
#n_neighbors：When constructing adjacency matrix with k-nearest neighbor method, the k-nearest neighbor is used；
#eigen_tol：Laplaces stop rule for Eigendecomposition of a matrix when using arpack eigen.
#assign_labels：By solving“the EIGENVALUES of Pierre-Simon Laplace Matrix” ,the indicator vector Matrix (similar to the feature of Reduced Dimension) is obtained, and the rows of the Matrix are clustered by using assign
sc.fit(X) 
label_pred = sc.labels_

# Draw the SpectralClustering result
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='group1')  
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='o', label='group2')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='o', label='group3')  
plt.legend(loc=2)  
plt.show()  
#Spectral clustering is a clustering method based on data similarity matrix,
#By introducing indicator variables, the partition problem is transformed into solving the optimal indicator variable matrix H.
#Then, by using the properties of Rayleigh entropy, the problem is further transformed into solving the K minimum eigenvalues of Laplacian matrix,
#Finally, the traditional clustering method is used for clustering.


# In[ ]:


#The second part is CLASSFICATION ANALYSIS
#I choose the DecisionTree to perform analysis


# In[48]:


#DecisionTree

#Each path from the root node to the leaf node of the decision tree builds a rule
#The characteristics of the intermediate nodes on the path correspond to the conditions of the rules,
#and the class labels of the leaf nodes correspond to the conclusions of the rules
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

iris_data = datasets.load_iris()
X=pca.fit_transform(x)
y = iris_data.target
label_list = ['group1', 'group2', 'group3']

#Building decision boundaries
def plot_decision_regions(X, y, classifier=None):
    marker_list = ['o', 'o', 'o']
    color_list = ['r', 'b', 'g']
    cmap = ListedColormap(color_list[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    t1 = np.linspace(x1_min, x1_max, 666)
    t2 = np.linspace(x2_min, x2_max, 666)

    #Grid drawing
    x1, x2 = np.meshgrid(t1, t2)
    #Call the predict function in the previous perception
    y_hat = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
    y_hat = y_hat.reshape(x1.shape)
    plt.contourf(x1, x2, y_hat, alpha=0.2, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for ind, clas in enumerate(np.unique(y)):
        plt.scatter(X[y == clas, 0], X[y == clas, 1], alpha=0.8, s=50,
                    c=color_list[ind], marker=marker_list[ind], label=label_list[clas])

#Training Module
tree = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
tree.fit(X, y)
    
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
        max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, presort=False, random_state=1,
        splitter='best')

plot_decision_regions(X, y, classifier=tree)

plt.legend()
plt.show()

#From this processus,we can see the the DecisionTree Algorithme has these features:
#As long as the tree goes down the tree root to the leaf, the split condition along the way can uniquely determine a classification predicate
#And suitable for high dimensional data
#But for the data with different sample size, the information gain tends to those with more values
#Easy over fitting


# In[ ]:




