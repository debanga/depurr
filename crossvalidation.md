[HOME](https://debanga.github.com/depurr)

# Cross Validation

## K Folds

```KFolds``` divides all the samples in ```k``` groups of samples, called folds. The machine learning model is trained using ```k-1``` folds, while the fold left out is used for validaton.

```
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state = 1)
for train_index, test_index in kf.split(X):
     X_train, y_train = X[train_index], y[train_index]
```

## Startified K Folds

## GroupK Folds

## Time Series Folds
