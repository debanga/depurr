[HOME](https://debanga.github.com/depurr)

# Cross Validation

## K Folds

KFold divides all the samples in k groups of samples, called folds ( if k=n this is equivalent to the Leave One Out strategy), of equal sizes (if possible). The prediction function is learned using k - 1 folds, and the fold left out is used for test.

<span style="color:red">```python```</span>
```
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state = 1)
for train_index, test_index in kf.split(X):
     X_train, y_train = X[train_index], y[train_index]
```

## Startified K Folds

## GroupK Folds

## Time Series Folds
