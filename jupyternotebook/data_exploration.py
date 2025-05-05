import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
sns.set()

#get all missing data
def get_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() * 100/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Missing', '%'])
    return missing_data

#heatmap of all features
def show_correlation(df):
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat);
    plt.show()

#target correlation matrix
def show_top_related(df, target, k):
    corrmat = df.corr()
    cols = corrmat.nlargest(k, target)[target].index
#     cm = np.corrcoef(df[cols].values.T)
#     plt.figure(figsize=(12,9))
#     sns.set(font_scale=1.25)
#     hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#     plt.show()
    return cols

#scatterplot of most related features
def show_pair_plot(df):
    sns.set()
    sns.pairplot(df, height = 2)
    plt.show();

#histogram and normal probability plot
def show_distribution(df, target):
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    sns.distplot(df[target], fit=norm);
    plt.subplot(122)
    res = stats.probplot(df[target], plot=plt)
    plt.show()

#boxplot of category feature and target    
def show_cat_relation(df, cat, target):
    concat = pd.concat([df[target], df[cat]], axis=1)
    f, ax = plt.subplots(figsize=(15, 6))
    fig = sns.boxplot(x=cat, y=target, data=concat)
    plt.xticks(rotation=90)
    plt.show()

#scatter plot of category feature and target    
def show_num_relation(df, num, target):
    data = pd.concat([df[target], df[num]], axis=1)
    data.plot.scatter(x=num, y=target);
    plt.show()