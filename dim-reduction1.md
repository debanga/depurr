# A Brief overview of Principal Component Analysis (PCA)

In this article, we discuss PCA and its related algorithms with example codes. In particular, we will include the following algorithms:
1. Principal Component Analysis
2. Linear Discriminent Analysis
3. Singular Value Decomposition

You may have seen these algorithms in different articles or books presented as separate concepts, but fundamentally they have many similarities. Studying them under one single lens of dimensionality reduction will provide us with a global understanding, and therefore, it will make it easier to remember these concepts.

---
### 💀 Only for coders, others can safely ignore! 

Import following packages in your Jupyter notebook:
 
```
# Basic packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
 
# Scikit-learn package
from sklearn.datasets import load_iris
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
```
---

## Data

We will use a practical hands-on approach to understand the algorithms. Let's get familiar with the simple yet popular [iris dataset](https://www.kaggle.com/arshid/iris-flower-dataset) that we are going to use in our illustrative examples.

The dataset description says,

> The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machines.
