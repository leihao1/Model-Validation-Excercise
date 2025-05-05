# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:08:34 2019

@author: TD ML Validation Team
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


#===============================================================================================
#pandas get_dummies to do one hot encoding
def OHE_get_dummies(df, categ_name):
    
    OHE_df = pd.get_dummies(df[categ_name], prefix=categ_name, drop_first = False)
    
    return OHE_df


#===============================================================================================
#return the feature importances of all the variables from RandomForest
def RandomForest_feature_selection(fe_name, columns_categ, matrix_x_input, label_y, seed, threshold = 0.0):
    
    # SelectfromModel
    clf = RandomForestClassifier(n_estimators=400, random_state=seed)
    clf.fit(matrix_x_input, label_y)
    sfm = SelectFromModel(clf, prefit=True, threshold=threshold)
    matrix_x_selected = sfm.transform(matrix_x_input)
    
    #print("Accuracy: ", clf.score(matrix_x_input, label_y))

    #================================
    feature_score_dict = {}
    for fn, s in zip(fe_name, clf.feature_importances_):
        feature_score_dict[fn] = s
        
    m = 0
    for k in feature_score_dict:
        if feature_score_dict[k] == 0.0:
            m += 1
    #print("number of not-zero features:" + str(len(feature_score_dict) - m))

    feature_score_dict_sorted = sorted(feature_score_dict.items(),
                                       key=lambda d: d[1], reverse=True)
#    print("feature_importance:")
#    for ii in range(len(feature_score_dict_sorted)):
#        print(feature_score_dict_sorted[ii][0], feature_score_dict_sorted[ii][1])
#    print("\n")

    f = open('./RandomForest_feature_importance.txt', 'w')
    f.write("Feature importance, threshold: " + str(threshold))
    f.write('\nRank\tFeature Name\tFeature Importance\n')
    for i in range(len(feature_score_dict_sorted)):
        f.write(str(i) + '\t' + str(feature_score_dict_sorted[i][0]) + '\t' + str(feature_score_dict_sorted[i][1]) + '\n')
    f.close()
    
#     #================================
#     how_long = matrix_x_select.shape[1] 
#     feature_used_dict_temp = feature_score_dict_sorted[:how_long]
#     feature_used_name = []
#     for ii in range(len(feature_used_dict_temp)):
#         feature_used_name.append(feature_used_dict_temp[ii][0])
#     print("feature_selected:\n")
#     for ii in range(len(feature_used_name)):
#         print(feature_used_name[ii])
#     print("\n")

#     f = open('./RandomForest_feature_selected.txt', 'w')
#     f.write('Feature Chose Name :\n')
#     for i in range(len(feature_used_name)):
#         f.write(str(feature_used_name[i]) + '\n')
#     f.close()
    
#     feature_not_used_name = []
#     for i in range(len(fe_name)):
#         if fe_name[i] not in feature_used_name:
#             feature_not_used_name.append(fe_name[i])

    #==============================================================
    # add the one-hot-ecoding feature importannce to one value
    # for the original catagorical variable
    
    feature_score_dict_new = feature_score_dict_sorted.copy()

    feature_score_dict_categ = []

    #columns_categ = ["JOB_TYPE", "SEX", "MARRIAGE", "EDUCATION"]

    for categ_name in columns_categ:

        feature_imp = 0.0

        for item in feature_score_dict_sorted:
            if item[0][0:len(categ_name)] == categ_name :
                feature_imp = feature_imp + item[1]

                feature_score_dict_new.remove(item)

        feature_score_dict_categ.append((categ_name, feature_imp))

    feature_score_dict_all = [*feature_score_dict_new, *feature_score_dict_categ]
    
    feature_score_dict_all_sorted = sorted(feature_score_dict_all, key=lambda d: d[1], reverse=True)


    return matrix_x_selected, feature_score_dict_all_sorted


#===============================================================================================
# provide the prefix of the categorical variable, and remove all the related one-hot-encoding
# variables from the column list
def remove_categ(categ_name, columns_X):
    
    columns_X_new = columns_X.copy()
    
    for item in columns_X:
        if item[0:len(categ_name)] == categ_name :
            columns_X_new.remove(item)
            
    return columns_X_new 

