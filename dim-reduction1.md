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

## Principal Component Analysis

Now, consider a classification task: given a vector of 4 feature values (in real numbers) ```X=[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]``` we have to predict what is the label ```y```, i.e. what is the species of flower it represents.  

Now, the question that arises is do we need all these 4 features to predict the species of flower, or we can do a "good" classification with a reduced number of features, say 2? That's where dimensionality reduction algorithm comes in handy. With a good dimensionality reduction algorithm we can reduce the number of features needed to perform a good classification. It has several benefits, such as:

1. Dropping less important features that do not significantly contribute to the prediction,
2. With less number of features we can perform predictions faster,
3. It allows to transform the existing features to new mutually independent features which have better predictive power

Of course, it comes with some disadvantages, such as, less interpretability of the transformed features, and loss of details from the data due to feature ellimination, but in practice in most of the cases, benefits from PCA trumps these disadvantages, and it is (or its variants) are widely used to solve real life problems. 

It's time to look at a brief fomulation of PCA, and it will follow an example of PCA with our Iris dataset.

### Theory


### Example
