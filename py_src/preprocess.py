import argparse
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import *
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import os
import sys
import numpy as np
import pandas as pd
import pdb

random_seed = None
l1_inverse = 1.0

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# ---------------------- Str 2 Integer
stage2id = {
    'Stage IA':  0, 
    'Stage IB':  1, 
    'Stage IIB': 2, 
    'Stage IIA': 3, 
    'Stage IV':  4,
    'Stage IIIA':5,
    'Stage IIIB':6,
    'IA':  0, 
    'IB':  1, 
    'IIB': 2, 
    'IIA': 3, 
    'IV':  4,
    'IIIA':5,
    'IIIB':6,
    'III' :5,
}

gen2id = {
    'Wild': 0, 
    'Mutation': 1,
}
# --------------------- dataset preprocess and transforms
def single_feature(data, column_name, dtype, missing_val, impute_strategy='mean', substitude_dict=None, normalize=False, one_hot=False,
                   delete=False,):
    """ replace the raw column and cast it to missing_val and imputing
    """
    if column_name not in data: 
        return data
    if delete == True:
        del data[column_name]
        return data

    if substitude_dict != None: 
        for raw_val, after_val in substitude_dict.items():
            data.loc[data[column_name]==raw_val, column_name] = after_val

    if dtype is not None: data[column_name] = data[column_name].astype(dtype)
    SimpleImputer(missing_values=missing_val, strategy=impute_strategy, fill_value=None, copy=False).fit_transform(data[[column_name]])

    if normalize: 
        scaler = StandardScaler()
        scaler = MinMaxScaler()
        if column_name == 'age': scaler = MinMaxScaler()
        data[column_name] = scaler.fit_transform(data[[column_name]])
    if one_hot  : 
        enc = OneHotEncoder()
        new_column = enc.fit_transform(data[[column_name]]).toarray()
        new_column_names = []
        for i in range(new_column.shape[1]):
            new_column_names.append( column_name + str(i) ) 
        df_features = pd.DataFrame(new_column, columns=new_column_names)    
        del data[column_name]
        data = data.join(df_features)
    return data

def process_y(data):
    df = data[['fustat', 'futime']]
    years = [1,3,5]
    values = []  # [1-year, 2-year, 3-year, 5-year]
    #import pdb
    #pdb.set_trace()
    for i in range(df.shape[0]):
        stat, time = df.loc[i]
        row = []
        for year in years:
            if time >= year * 365: row.append(0)
            elif stat == 1 and time < year * 365: row.append(1)
            elif stat == 0 and time < year * 365: 
                row.append(np.NaN)
        values.append(row)
    column_name = ['year_'+str(i) for i in years]
    year_survival_rate = pd.DataFrame(values, index=data.index, columns=column_name)
    data = data.join(year_survival_rate)
    #print (data[['fustat', 'futime']+column_name])
    imp = IterativeImputer(max_iter=20, random_state=random_seed, min_value=0, max_value=1)
    imp.fit(data)
    arr_values = (imp.transform(data))
    arr_values = np.round(arr_values)
    data[column_name] = pd.DataFrame(arr_values, columns=data.columns)[column_name]
    del data['fustat']
    del data['futime']
    return data, column_name

# start classification
names = [
         "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", 
         "Ridge",
         "Perception",
         "LogisticRegression",
         "MultinomialNB", 
        ]

classifiers = {
    "NearestNeighbor": KNeighborsClassifier(5),
    "LinearSVM":SVC(kernel="linear", C=0.025),
    "RBFSVM": SVC(gamma=2, C=1),
    "GaussiaProcess":GaussianProcessClassifier(1.0 * RBF(1.0)),
    "DecisionTree":DecisionTreeClassifier(max_depth=5),
    "RandomForest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "MLP": MLPClassifier(alpha=1, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "NaiveBayes":GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(), 
    "Ridge": RidgeClassifier(), 
    "Perceptron": Perceptron(), 
    "LogisticRegression": LogisticRegression(solver="liblinear", random_state=random_seed), 
    "Lasso": LogisticRegression(solver="liblinear", random_state=random_seed, penalty="l1", C=l1_inverse), 
    "MultinomialNB": MultinomialNB(),
}

RegressionMethod = {
    'Lasso': Lasso() ,
}


# ---------- start classify
def cal_metrics(y_test, y_pred):
    TP = ((y_test == y_pred) & (y_pred == 1.0)).sum()
    TN = ((y_test == y_pred) & (y_pred == 0.0)).sum()
    FN = ((y_test != y_pred) & (y_pred == 0.0)).sum()
    FP = ((y_test != y_pred) & (y_pred == 1.0)).sum()
    return [
            1.0*TP/(TP+FN),  # 敏感性
            1.0*TN/(TN+FP),  # 特异性
            1.0*TP/(TP+FP),  # 精确度
            1.0*TP/(TP+FN),  # 召回率
            1.0*(TP+TN)/(TP+FN+TN+FP), #准确度
           ]

def print_metric(clf_name, fitted_clf, y_test, y_pred, y_prob=None):
    indent = 0
    print ('\t'*indent + clf_name, ":")
    indent += 1
    metrics = cal_metrics(y_test, y_pred)
    print ('\t'*indent + 'Accuracy   :\t', metrics[4])
    print ('\t'*indent + 'Sensitivity:\t', metrics[0])
    print ('\t'*indent + 'Specificity:\t', metrics[1])
    print ('\t'*indent + 'Precise    :\t', metrics[2])
    print ('\t'*indent + 'Recall     :\t', metrics[3])
    print ('\t'*indent + 'F1         :\t', 2.0/(1.0/metrics[2]+1.0/metrics[3]))
    if y_prob is not None: #hasattr(fitted_clf, 'predict_proba'):
        print ('\t'*indent + "AUC:        \t", roc_auc_score(y_test, y_prob))

def experiment(dataset, y_label, x_labels=None, test_set='test'):
    raw_x_train, raw_y_train, raw_x_test ,raw_y_test = dataset

    y_train, y_test = raw_y_train[y_label], raw_y_test[y_label]
    x_train, x_test = raw_x_train,          raw_x_test
    if x_labels is not None:
        x_train = raw_x_train[x_labels]
        x_test  = raw_x_test [x_labels]
    if test_set == 'train':
        """ use the trainset as testset
        """
        y_test = y_train
        x_test = x_train

    print("Feature of X({}):".format(x_train.columns.__len__()), x_train.columns)
    print("Zero / One:\t{}/{}".format((y_train==0).sum(), (y_train==1).sum()))
    ros = RandomOverSampler(random_state=0)
    x_train, y_train = ros.fit_resample(x_train, y_train)
    print ("Resampled dataset:{}".format(Counter(y_train)))

    for clf_name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_prob = None
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(x_test)[:,1]
        print_metric(clf_name, clf, y_test, y_pred, y_prob)
    
    if is_output: 
        
        assert len(classifiers) == 1, "classifiers can't be None because of output flag is True"
        output_test_pd = raw_x_test.copy(deep=True)
        output_test_pd = output_test_pd.join(y_test)
        output_test_pd = output_test_pd.join(
            pd.DataFrame({"output_{}".format(y_label):np.reshape(y_pred, [-1]), "output_{}_prob".format(y_label):np.reshape(y_prob, [-1])}, 
                        index=output_test_pd.index)
        )
        output_test_pd.to_csv("./output/output.csv")

    if y_label == 'year_5':
        ...
        #import pdb
        #pdb.set_trace()

def start_experiment(input_file, output_file, ylabel, xlabels, interested_classifier, train_test_ratio=0.9):
    if interested_classifier is not None:
        classifiers = {interested_classifier: classifiers[interested_classifier]}

    data = pd.read_cvs(input_file)
    data_train, data_test = train_test_split(data, train_size=train_test_ratio)
    raw_y_traig = data_train[y_labels]
    raw_x_train = data_train.drop(y_labels, axis=1)
    from collections import Counter
    raw_y_test = data_test[y_labels]
    raw_x_test = data_test.drop(y_labels, axis=1)
    experiment((raw_x_train, raw_y_train, raw_x_test, raw_y_test, y_label, xlabels, 'test'), 

