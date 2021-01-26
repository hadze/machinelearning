from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()

# Single metric evaluation using cross_validate
cv_results = cross_validate(lasso, X, y, cv=3)
sorted(cv_results.keys())
#['fit_time', 'score_time','test_score']
cv_results['test_score']

# Multiple metric evaluation using cross_validate
# Refer the scoring parameter doc for more information
scores = cross_validate(lasso, X, y, cv=3, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
print(scores['test_neg_mean_squared_error'])
print(scores['train_r2'])