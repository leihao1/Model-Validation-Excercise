import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

#replace question marks with np.nan
def replace_question_marks(df):
    try:
        df = df.replace({'?' : np.nan})
        print("Replaced all '?' to np.nan")
    except:
        print('No question marks found')
    return df

#check whether dataset is balanced
def check_class_distribution(df):
    print('Class distributions:')
    print(df.iloc[:,-1].value_counts())
    
#PCA dimension reducetion
def dimension_reduction(x_train, x_test, n_components, verbose=True, upper_bound=0):
    embedding = None
    if x_train.shape[1] >= upper_bound:
        if verbose:
            print('Reduce dimension form %s to %s'%(x_train.shape[1],n_components))
        pca = PCA(n_components=n_components, random_state=33)
        pca.fit(x_train)
        x_train = pd.DataFrame(pca.transform(x_train))
        x_test = pd.DataFrame(pca.transform(x_test))
        embedding = pca
    return x_train, x_test

#convert string value to integer(ignore missing data)
# def encode_labels(x_train, x_test, encoder):
#     df = pd.concat([x_train,x_test],axis=0) 
#     #encoding y labels
#     if len(x_train.shape)==1:
#         print('Encoding y label values...')
#         not_null_df = df[df.notnull()]
#         encoder.fit(not_null_df)
#         x_train = encoder.transform(x_train)
#         x_test = encoder.transform(x_test)
#     #encoding x features
#     else:
#         print('Encoding X features...')
#         for i,t in enumerate(df.dtypes):
#             if t == 'object':
#                 s_df = df.iloc[:,i]
#                 not_null_df = s_df.loc[s_df.notnull()]
#                 encoder.fit(not_null_df)
#                 try:
#                     x_train.iloc[:,i] = x_train.iloc[:,i].astype('float')
#                 except:
#                     x_train.iloc[:,i] = x_train.iloc[:,i].apply(lambda x: encoder.transform([x])[0] if x not in [np.nan] else x)
#                 try:
#                     x_test.iloc[:,i] = x_test.iloc[:,i].astype('float')
#                 except:
#                     x_test.iloc[:,i] = x_test.iloc[:,i].apply(lambda x: encoder.transform([x])[0] if x not in [np.nan] else x) #np.nan
#     return x_train, x_test

def encode_labels(x_train, x_test, encoder=preprocessing.LabelEncoder()):
    df = pd.concat([x_train,x_test],axis=0)
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    #encoding y labels
    if len(x_train.shape)==1:
        print('Encoding y label values...')
        not_null_df = df[df.notnull()]
        encoder.fit(not_null_df)
        x_train = encoder.transform(x_train)
        x_test = encoder.transform(x_test)
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
    #encoding x features
    else:
        print('Encoding X features...')
        for i,t in enumerate(df.dtypes):
#             print(df.shape)
            if t == 'object':
                series = df.iloc[:,i]
                not_null_df = series.loc[series.notnull()]
                encoder.fit(not_null_df)
                try:
                    x_train.iloc[:,i] = x_train.iloc[:,i].astype('float')
                except:
                    not_nan_indexs = x_train[x_train.iloc[:,i].notna()].index
#                     not_nan_indexs = [not_nan_indexs.tolist().index(i) for i in not_nan_indexs.tolist()]
                    x_train.iloc[not_nan_indexs,i] = encoder.transform(x_train.iloc[not_nan_indexs,i])
                try:
                    x_test.iloc[:,i] = x_test.iloc[:,i].astype('float')
                except:
                    not_nan_indexs = x_test[x_test.iloc[:,i].notna()].index
#                     not_nan_indexs = [not_nan_indexs.tolist().index(i) for i in not_nan_indexs.tolist()]
                    x_test.iloc[not_nan_indexs,i] = encoder.transform(x_test.iloc[not_nan_indexs,i])
    return x_train, x_test

#put class colunmn at end of dataframe
def reorder_columns(dataFrame, target_column):
    target_column += 1
    cols = dataFrame.columns.tolist()
    cols = cols[target_column:] + cols[:target_column]
    return dataFrame[cols]

#impute missing data with given strategy
def impute_value(x_train, x_test, strategy):
    if strategy == 'drop' or strategy == None:
        return x_train.dropna(), x_test.dropna()
    else:
        print('Imputed missing data with "%s"'%strategy)
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        train_type_dic = dict()#keep original train data type before impute
        columns = x_train.columns
        for i,t in enumerate(x_train.dtypes):
            if t != 'object':
                train_type_dic[i] = t
        test_type_dic = dict()#keep original test data type before impute
        for i,t in enumerate(x_test.dtypes):
            if t != 'object':
                test_type_dic[i] = t
        x_train = pd.DataFrame(imp.fit_transform(x_train))
        x_test = pd.DataFrame(imp.transform(x_test))
        x_train.columns = columns
        x_test.columns = columns
#         for key in train_type_dic:
#             x_train.iloc[:,key] = x_train.iloc[:,key].astype(train_type_dic[key])
#         for key in test_type_dic:
#             x_test.iloc[:,key] = x_test.iloc[:,key].astype(test_type_dic[key])
    assert x_train.isnull().sum().max() == 0
    assert x_test.isnull().sum().max() == 0 
    return x_train, x_test
    
# standardize data with given scaler, default: StandardScaler
def standardize_data(x_train, x_test, scaler):
    print('Standardized data with %s'%str(type(scaler)).split('.')[-1].split("'>")[0])
    columns = x_train.columns
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    x_train.columns = columns
    x_test.columns = columns
    return x_train, x_test

#drop feature columns whose missing value ratio is bigger than threshold
def drop_features(x_train, x_test, threshold):
    total = x_train.isnull().sum().sort_values(ascending=False)
    percent = (x_train.isnull().sum()/x_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
    indexes = (missing_data[missing_data['%'] > threshold]).index
    x_train = x_train.drop(indexes, 1)
    x_test = x_test.loc[:,x_train.columns]
    print('Dropped %s features'%len(indexes))
    return x_train, x_test

#transform to gaussion distribution
def gaussion_normarlize(x_train, x_test):
    print('Normalized to gaussion distribution')
    columns = x_train.columns
    pt = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
    # pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    x_train = pt.fit_transform(x_train)
    x_test = pt.transform(x_test)
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    x_train.columns = columns
    x_test.columns = columns
    return x_train, x_test
