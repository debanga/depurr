# A Practical Guide to Principal Component Analysis (PCA)

![](https://c1.wallpaperflare.com/preview/270/321/283/block-chain-data-records-concept-system-communication.jpg)

In this article, we discuss PCA and its related algorithm Singular Value Decomposition (SVD) with example codes. Looking at PCE and SVD under one single lens of dimensionality reduction will provide us with a global understanding, and therefore, it will make it easier to remember these concepts.

Please note that there are different variants of PCA (e.g. probabilistic PCA) that can outperform vanilla PCA, that we will discuss in this article, in many occasions. But, my goal is not to explore PCA exhaustively, rather keep the discussion beginner friendly. I can discuss other advance variants in future articles.



# Data

We will use a practical hands-on approach to understand the algorithms. Let's get familiar with the simple yet popular [iris dataset](https://www.kaggle.com/arshid/iris-flower-dataset) that we are going to use in our illustrative examples.

The dataset description says,
> The Iris dataset was used in R.A. Fisher's classic 1936 paper, "[The Use of Multiple Measurements in Taxonomic Problems](http://rcs.chemometrics.ru/Tutorials/classification/Fisher.pdf)", and can also be found on the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/). It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machines.

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris.png)

The data set consists of 50 samples from each of three species of Iris: **Iris Setosa, Iris virginica, and Iris versicolor.**. Therefore, each sample will have a ```label``` from one of these three species.

Four ```features``` were measured from each sample: the length and the width of the sepals and petals, in centimeters. These features are: **Sepal Length, Sepal Width, Petal Length, and Petal Width.**

Here is a peek at the data in tabular format:

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-table.png)

Statistical distribution of the features are:

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-dist.png)

Now, we have a general idea of the dataset, and we are ready to use it in our future discussions.

# Principal Component Analysis

Now, consider a classification task: given a vector of 4 feature values (in real numbers) ```x=[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]``` we have to predict what is the label ```y```, i.e. what is the species of flower it represents.  

Now, the question that arises is do we need all these 4 features to predict the species of flower, or we can do a "good" classification with a reduced number of features, say 2? That's where dimensionality reduction algorithm comes in handy. With a good dimensionality reduction algorithm we can reduce the number of features needed to perform a good classification. It has several benefits, such as:

1. Dropping less important features that do not significantly contribute to the prediction,
2. With less number of features we can perform predictions faster,
3. It allows to transform the existing features to new mutually independent features which have better predictive power

Of course, it comes with some disadvantages, such as, less interpretability of the transformed features, and loss of details from the data due to feature ellimination, but in practice in most of the cases, benefits from PCA trumps these disadvantages, and it is (or its variants) are widely used to solve real-world problems. 

It's time to look at a brief fomulation of PCA, and it will follow an example of PCA with our Iris dataset.


## Theory
PCA belongs to the ```unsupervised machine learning algorithms```. In unsupervised algorithms, we try to understand the data without looking at the labels, instead we analyze the distribution of data and recognize patterns, such as finding clusters and orthogonal features. 

As we mentioned before, the goal of PCA is to reduce the number of features by projecting them into a reduced space constructed my mutually orthogonal features (also known as "principal components") with a compact represention of the distribution of data.


We can break down the compuation of PCA into several steps:

### **Generation of feature matrix:** 

Since PCA is an unsupervised algorithm, we do not use the labels, but only the features. Let's consider $X$ is the feature matrix of size $n$ x $m$, where $n$ is the total number of samples and $m$ is the number of features. In our Iris Dataset example $n=150$ and $m=4$.

Therefore, $X=$ 

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-x.png)

### **[Data standarization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html):** 

To remove the sensitivity of the variance of the feature values, we standardize features by removing the mean and scaling to unit variance as $X_{standadized}=(X-X_{m})/X_{v}$, where, $X_{standadized}$ is the standardized feature matrix, $X_m$ is the feature wise mean and $X_v$ is feature wise standard deviation.

Therefore, $X_{standadized}=$

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-standard.png)


### **[Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD):** 

Now, we need to find the eignevalues and eigenvectors of the dataset. There are different ways of doing it, such as (1) eignevalue decomposition of the covariance matrix of the data, (2) singular value decomposition, etc. All of them essentially give the same result, but considering the computational efficiency of SVD, and also, to remain aligned with our original goal to relate PCA with other related concepts, we will show the application of SVD here. 

If we now perform singular value decomposition of $\mathbf X$, we obtain a decomposition 

$$\mathbf X = \mathbf U \mathbf S \mathbf V^\top,$$ 

where $\mathbf U$ is a unitary matrix and $\mathbf S$ is the diagonal matrix of singular values $s_i$. 

From here one can easily see that $$\mathbf C = \mathbf V \mathbf S \mathbf U^\top \mathbf U \mathbf S \mathbf V^\top /(n-1) = \mathbf V \frac{\mathbf S^2}{n-1}\mathbf V^\top,$$ meaning that right singular vectors $\mathbf V$ are principal directions and that singular values are related to the eigenvalues of covariance matrix via $\lambda_i = s_i^2/(n-1)$. 

Now, if we perform singular value decomposition of $X$, we will get:

$V$ is a 4x4 matrix, and represent the eigenvectors:

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-svd-u.png)

$S$ is a 4x150 matrix, and 4 diagonal components are represented by (say, $S_{diag}$):

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-svd-s.png)

$U$ is a 150x150 matrix:

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-svd-v.png)

The eigenvalues are estimated as: ```[2.93035378, 0.92740362, 0.14834223, 0.02074601]```


### **Drop eigenvectors to reduce dimensionality:** 

We sort the eigenvectors by decreasing eigenvalues (or singular values) and choose $k$ eigenvectors with the largest eigenvalues to form a $m$ × $k$ dimensional matrix $P$. This matrix $P$ can be called a "projection matrix" and it can now be used to sample new points with 4 features into the reduced space with only $k$ dimensions.

If we use $k=2$; our singular values are: 

```[2.93035378, 0.92740362]```

and corresponding eigen vectors are

```[-0.522372, 0.263355, -0.581254, -0.565611]``` and ```[-0.372318,-0.925556,-0.021095,-0.065416]```.

Therefore, $P=$

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-projection.png)

### **Project features to reduced space**

Now, using $P$ we can project our original 4 feature data, $X$ to a reduced 2 dimentional space as $X_{reduced}$:

$$X_{reduced} = X \times P$$

and $X_{reduced}$=

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris_reduced.png)


## Visualize new features

Now, let's visualize the samples in reduced 2 dimensional space and see if PCA has been able to separate different iris species.

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-plot.png)

As we can see, even after reducing the number of features from 4 to 2, the new features can separate the iris species in the form of separate clusters.


## Our analysis can be done just in a few lines using scikit-learn package

Fortunately, in future we don't have to do all the above analysis, as ```PCA``` function is available in ```scikit-learn``` python package. After data standardization step we can simply call the ```PCA``` function as follows to get $X_{reduced}$:

```
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_standardized)
```

After plotting the results we can see that we get the same results as before:

![](https://raw.githubusercontent.com/debanga/depurr/master/images/iris-reduced-scikit.png)


# Code

```
# Basic packages
import numpy as np
import pandas as pd 
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# Scikit-learn package
from sklearn.preprocessing import StandardScaler

# Import data
dataset = pd.read_csv('https://raw.githubusercontent.com/debanga/depurr/master/datasets/Iris.csv').drop(columns=['Id'])

# See a tabular sample
dataset.head()

# Get feature distribution
dataset.describe()

# Generate X and y
X = dataset.drop(columns=['Species'])
y = dataset['Species']

# Standardization
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X = X.rename(columns={0:'SepalLengthCm',1:'SepalWidthCm',2:'PetalLengthCm',3:'PetalWidthCm'})

# Singular Value Decoposition
u,s,v = np.linalg.svd(X)

# Estimate singular values
singular_values = s*s/(X.shape[0]-1)

# Top k=2 singular values and corresponding eigenvectors
k = 2
print(f"Top {k} eigen values:")
print(singular_values[:k])

print(f"Top {k} eigen vectors:")
print(v.T[:,0])
print(v.T[:,1])

print('Projection matrix is: ')
print(v.T[:,:2])

# Data in reduced dimension
X_reduced = np.matmul(np.array(X),v.T[:,:2])
pd.DataFrame(X_reduced)

# Visualize the samples in reduced space
dataset_new = pd.concat([pd.DataFrame(X_reduced),pd.DataFrame(dataset['Species'])], axis=1)
dataset_new = dataset_new.rename(columns={0:"feature_1",1:"feature_2"})
ax = sns.scatterplot(x="feature_1", y="feature_2", hue="Species", data=dataset_new)
plt.show()

print('Plot of principal components estimated using scikit-learn')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced_scikit = pca.fit_transform(X)
dataset_new_scikit = pd.concat([pd.DataFrame(X_reduced_scikit),pd.DataFrame(dataset['Species'])], axis=1)
dataset_new_scikit = dataset_new.rename(columns={0:"feature_1",1:"feature_2"})
ax = sns.scatterplot(x="feature_1", y="feature_2", hue="Species", data=dataset_new_scikit)
plt.show()
```

# Jupyter Notebook

![](https://www.kaggle.com/debanga/a-practical-look-at-principal-component-analysis)

## References
[1] http://suruchifialoke.com/2016-10-13-machine-learning-tutorial-iris-classification/

[2] https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643

[3] https://www.kaggle.com/vipulgandhi/pca-beginner-s-guide-to-dimensionality-reduction

[4] https://stats.stackexchange.com/posts/134283
