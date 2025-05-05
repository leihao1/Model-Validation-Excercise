import time
import scipy
from scipy import stats
from scipy.stats import norm

import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV

K = None
# classifiers
def train_KNN(X_train, y_train):
    print('Training KNN ...')
    knn = KNeighborsClassifier()
#     scoring = ['roc_auc']
    param_distributions = {
        'n_neighbors' : scipy.stats.randint(1,20)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        knn,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

def train_SVM(X_train, y_train):
    print('Training SVM ...')
    svm = SVC(kernel='rbf', probability=True, cache_size=3000, random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'C' : scipy.stats.reciprocal(1.0, 100.),
        'gamma' : scipy.stats.reciprocal(0.01, 10.),
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        svm,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

def train_DecisionTree(X_train, y_train):
    print('Training DecisionTree ...')
    tree = DecisionTreeClassifier(random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'max_depth' : scipy.stats.randint(10,1000)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        tree,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

def train_RandomForest(X_train, y_train):
    print('Training RandomForest ...')
    forest = RandomForestClassifier(random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'max_depth' : scipy.stats.randint(10,100),
        'n_estimators' : scipy.stats.randint(100,1000)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        forest,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

def train_AdaBoost(X_train, y_train):
    print('Training AdaBoost ...')
    boost = AdaBoostClassifier(random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'learning_rate' : scipy.stats.uniform(0.75, 1.25),
        'n_estimators' : scipy.stats.randint(40,70)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        boost,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_    

def train_LogisticRegression(X_train, y_train):
    print('Training LogisticRegression ...')
    lr = LogisticRegression(solver='liblinear', multi_class='auto', random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'C' : scipy.stats.reciprocal(1.0, 1000.),
        'max_iter' : scipy.stats.randint(100,1000)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        lr,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_ 

def train_GaussianNaiveBayes(X_train, y_train):
    print('Training GaussianNaiveBayes ...')
    gaussian = GaussianNB()
#     scoring = ['roc_auc']
    param_distributions = {
        'var_smoothing' : scipy.stats.uniform(1e-10, 1e-9),
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        gaussian,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_ 

def train_NeuralNetwork(X_train, y_train):
    print('Training NeuralNetwork ...')
    nn = MLPClassifier(solver='adam', random_state=0)
#     scoring = ['roc_auc']
    param_distributions = {
        'hidden_layer_sizes' : [(100,50,10)],
        'learning_rate_init' : scipy.stats.uniform(0.001, 0.005),
        'max_iter' : scipy.stats.randint(200,500)
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        nn,param_distributions,n_iter=5,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_

def train_XGBoost(X_train, y_train):
    from xgboost import XGBClassifier, plot_importance
    print('Training XGBoost ...')
    xgb = XGBClassifier()
    param_distributions = {
        "objective": ['binary:logistic'],
        "n_estimators": scipy.stats.randint(50,100),
        "max_depth": scipy.stats.randint(3,10), 
        "colsample_bylevel": [0.5, 0.7, 1]
    }
    randcv = sklearn.model_selection.RandomizedSearchCV(
        xgb,param_distributions,n_iter=10,cv=3,n_jobs=K,random_state=0)
    randcv.fit(X_train, y_train)
    return randcv.best_estimator_ 

#run all models on training set and return all the bests
def run_all_clfs(X_train, y_train, X_test, y_test):
    all_clfs = []
    clf_names = []
    clf_times = []
    start = time.time()
    
    clf1 = train_KNN(X_train, y_train)
    all_clfs.append(clf1)
    clf_names.append('KNN')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    clf2 = train_SVM(X_train, y_train)
    all_clfs.append(clf2)
    clf_names.append('SVM')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    clf3 = train_DecisionTree(X_train, y_train)
    all_clfs.append(clf3)
    clf_names.append('Decision Tree')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    clf4 = train_RandomForest(X_train, y_train)
    all_clfs.append(clf4)
    clf_names.append('Random Forest')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    clf5 = train_AdaBoost(X_train, y_train)
    all_clfs.append(clf5)
    clf_names.append('AdaBoost')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    clf6 = train_LogisticRegression(X_train, y_train)
    all_clfs.append(clf6)
    clf_names.append('Logistic regression')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    clf7 = train_GaussianNaiveBayes(X_train, y_train)
    all_clfs.append(clf7)
    clf_names.append('Gaussian Naive Bayes')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    clf8 = train_NeuralNetwork(X_train, y_train)
    all_clfs.append(clf8)
    clf_names.append('NeuralNetwork')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    return all_clfs, clf_names, clf_times

#only run Logistic Regression for demo
def run_lr_clf(X_train, y_train, X_test, y_test):
    all_clfs = []
    clf_names = []
    clf_times = []
    start = time.time()
    
    clf6 = train_LogisticRegression(X_train, y_train)
    all_clfs.append(clf6)
    clf_names.append('Logistic regression')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    return all_clfs, clf_names, clf_times

#only run Logistic Regression for demo
def run_xgb_clf(X_train, y_train, X_test, y_test):
    all_clfs = []
    clf_names = []
    clf_times = []
    start = time.time()
    
    clf6 = train_XGBoost(X_train, y_train)
    all_clfs.append(clf6)
    clf_names.append('XGBoost')
    end = time.time()
    clf_times.append(end - start)
    start = end
    
    return all_clfs, clf_names, clf_times