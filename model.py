import pandas as pd
import numpy as np
import sklearn.linear_model as linear
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.ensemble as esmb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#Linear Regression
def linearRegressor(datasource):
    dataset = pd.read_csv("./dataset/"+datasource)
    dataset = dataset.dropna()
    X = dataset[["Open", "High", "Low", "Volume"]]
    y = dataset[["Close"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = linear.LinearRegression()
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_test)
    a = np.array(y_test)
    b = np.array(prediction)
    df = pd.DataFrame({"Actual": a.flatten(), "Predicted": b.flatten()})
    print(df)
    print("MAE : ", metrics.mean_absolute_error(a.flatten(), b.flatten()))
    #print("MSE : ", metrics.mean_squared_error(a.flatten(), b.flatten()))
    #print("RMSE : ", np.sqrt(metrics.mean_squared_error(a.flatten(), b.flatten())))
    print("R2 score : ", metrics.r2_score(a.flatten(), b.flatten()))
    #print(valid_scores)

    ##### Cross Validating #####
    plt.figure()
    plt.title("Linear Regression Cross Validation")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores  = learning_curve(linear.LinearRegression(), X_train, y_train, cv=cv, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    
    
#Random Forest
def randForestRegressor(datasource):
    dataset = pd.read_csv("./dataset/"+datasource)
    dataset = dataset.dropna()
    X = dataset[["Open", "High", "Low", "Volume"]]
    y = dataset[["Close"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = esmb.RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
    np_train = np.array(y_train)
    model.fit(X_train, np_train.ravel())

    prediction = model.predict(X_test)
    a = np.array(y_test)
    b = np.array(prediction)
    df = pd.DataFrame({"Actual": a.flatten(), "Predicted": b.flatten()})
    print(df)
    print("MAE : ", metrics.mean_absolute_error(a.flatten(), b.flatten()))
    #print("MSE : ", metrics.mean_squared_error(a.flatten(), b.flatten()))
    #print("RMSE : ", np.sqrt(metrics.mean_squared_error(a.flatten(), b.flatten())))
    print("R2 score : ", metrics.r2_score(a.flatten(), b.flatten()))

    ##### Cross Validating #####
    plt.figure()
    plt.title("Random Forest Cross Validation")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores  = learning_curve(esmb.RandomForestRegressor(n_estimators=100, random_state=0), X_train, np_train.ravel(), cv=cv, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

#SVM
def SVM(datasource):
    dataset = pd.read_csv("./dataset/"+datasource)
    dataset = dataset.dropna()
    X = dataset[["Open", "High", "Low", "Volume"]]
    y = dataset[["Close"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    y_train = np.array(y_train).ravel()
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    model = svm.SVR(kernel='sigmoid', degree=1, gamma='auto', tol=.1, C=16.2,
                    epsilon=0.1, shrinking=True, max_iter=10000, coef0=1e-5)
    model.fit(X_train, y_train)

    X_test = scaler.fit_transform(X_test, y_test)
    prediction = model.predict(X_test)
    a = np.array(y_test)
    b = np.array(prediction)
    df = pd.DataFrame({"Actual": a.flatten(), "Predicted": b.flatten()})
    print(df)
    print("MAE : ", metrics.mean_absolute_error(a.flatten(), b.flatten()))
    #print("MSE : ", metrics.mean_squared_error(a.flatten(), b.flatten()))
    #print("RMSE : ", np.sqrt(metrics.mean_squared_error(a.flatten(), b.flatten())))
    print("R2 score : ", metrics.r2_score(a.flatten(), b.flatten()))
    #print(valid_scores)

    ##### Cross Validating #####
    model2 = svm.SVR(kernel='sigmoid', degree=1, gamma='auto', tol=.1, C=16.2,
                    epsilon=0.1, shrinking=True, max_iter=10000, coef0=1e-5)
    plt.figure()
    plt.title("SVM Cross Validation")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores  = learning_curve(model2, X_train, y_train, cv=cv, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
