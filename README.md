[scikit-learn](https://scikit-learn.org/stable/) framework based prediction model

This model predicts the close stock price using datasets from yahoo finance

Requirement:
  1. [python 3.6.5](https://www.python.org/downloads/release/python-365/)
  2. [pandas](https://pandas.pydata.org/)
  3. [scikit-learn](https://scikit-learn.org/stable/)
  4. [numpy](http://www.numpy.org/)
  5. [matplotlib](https://matplotlib.org/)

How to use:
  1. Run 'executable.py'
  2. Choose dataset
  3. Choose the algorithm
  4. The results should pop up after a few minutes (MAE and Cross Validation)
 
 Algorithm options :
  1. Simple Linear Regression. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
  2. Random Forest Regressor. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
  3. Support Vector Machine Regressor (The predictor model uses sigmoid as its kernel). https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
  
To tweak the model, you can modified the model.py file and change its parameter before fitting data into the model.
