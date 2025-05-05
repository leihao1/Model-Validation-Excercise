import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.preprocessing import label_binarize
from pre_processing import dimension_reduction
import seaborn as sns
sns.set()
"""
Evaluate different models and plot metrics
- regression: MSE, R-Square
- classification: ROC, PR
"""
def evaluate_regressors(X_train, y_train, X_test, y_test, all_regrs, regr_names, file_name, output_path):
    plt.figure(figsize=(18,8))
    plt.suptitle("Dataset: %s"%file_name, size=16)
    ax1 = plt.subplot(121)
    mse_scores = plot_mse_score(X_train, y_train, X_test, y_test, all_regrs, regr_names, ax1)
    ax2 = plt.subplot(122)
    ax2.set_xlim(0,1)
    r2_scores = plot_r2_score(X_train, y_train, X_test, y_test, all_regrs, regr_names,ax2)
    plt.savefig(output_path+file_name.split('.')[0]+'_mse-r2')
    plt.show()
    return mse_scores,r2_scores

def plot_mse_score(X_train, y_train, X_test, y_test, all_regrs, regr_names, ax):
    mse_scores = dict()
    training_scores = []
    test_scores = []
    
    for regr, regr_name in zip(all_regrs, regr_names):
        train_preds = regr.predict(X_train)
        test_preds = regr.predict(X_test)
        train_score = sklearn.metrics.mean_squared_error(y_train, train_preds)
        test_score = sklearn.metrics.mean_squared_error(y_test, test_preds)
        training_scores.append(train_score)
        test_scores.append(test_score)
        mse_scores[regr_name] = test_score
        
    N = len(all_regrs)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.barh(ind-width/2, training_scores, align='center', label='Training Set', height=width)
    p2 = plt.barh(ind+width/2, test_scores, align='center', label='Test Set', height=width)
    for i, v in enumerate(training_scores):
        plt.text(v,ind[i]-width/2.5,'%.3f'%v)
        plt.text(test_scores[i],ind[i]+width/1.5,'%.3f'%test_scores[i])
        
    plt.yticks(ind, regr_names) 
    plt.xlabel('MSE')
    plt.title('Mean Squared Error Of All Regressors')
    plt.legend(handles=[p1,p2], loc='upper left')
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
#     ax.yaxis.tick_right()
    return mse_scores

def plot_r2_score(X_train, y_train, X_test, y_test, all_regrs, regr_names, ax):
    r2_scores = dict()
    training_scores = []
    test_scores = []
    
    for regr, regr_name in zip(all_regrs, regr_names):
        train_preds = regr.predict(X_train)
        test_preds = regr.predict(X_test)
        train_score = sklearn.metrics.r2_score(y_train, train_preds)
        test_score = sklearn.metrics.r2_score(y_test, test_preds)
        training_scores.append(train_score)
        test_scores.append(test_score)
        r2_scores[regr_name] = test_score
        
    N = len(all_regrs)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

#     p1 = plt.bar(ind, training_scores, width)
#     p2 = plt.bar(ind+width, test_scores, width)
#     plt.ylabel('Scores')
#     plt.title('Scores by group and gender')
#     plt.xticks(ind, regr_names,rotation='vertical')
#     plt.yticks(np.arange(0, 1.1, 0.1))
#     plt.legend((p1[0], p2[0]), ('Training', 'Test'))

    p1 = plt.barh(ind-width/2, training_scores, align='center', label='Training Set', height=width)
    p2 = plt.barh(ind+width/2, test_scores, align='center', label='Test Set', height=width)
    for i, v in enumerate(training_scores):
        plt.text(v+0.01,ind[i]-width/2.5,'%.3f'%v)
        plt.text(max(test_scores[i],0)+0.01,ind[i]+width/1.5,'%.3f'%test_scores[i])

    plt.yticks(ind, regr_names)
    plt.xlabel('R² Score')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('R² Scores Of All Regressors')
    plt.legend(handles=[p1,p2], loc='upper right')
    plt.gca().invert_yaxis()
#     plt.gca().invert_xaxis()
    ax.yaxis.tick_right()
    return r2_scores


def evaluate_classifiers(X_test, y_test, all_clfs, clf_names, file_name=None, output_path=None):
    plt.figure(figsize=(16,8))
    if file_name:
        plt.suptitle("Dataset: %s"%file_name, size=16)
    plt.subplot(121)
    roc_scores = plot_roc_curve(X_test, y_test, all_clfs, clf_names)
    plt.subplot(122)
    pr_scores = plot_pr_curve(X_test, y_test, all_clfs, clf_names)
    if output_path:
        plt.savefig(output_path+file_name.split('.')[0]+'-roc_pr')
    plt.show()
    return roc_scores, pr_scores

#plot tools
def plot_roc_curve(X_test, y_test, all_clfs, clf_names):
    if type(X_test) is list:
        return plot_roc_curve2(X_test, y_test, all_clfs, clf_names)
    roc_scores = dict()
#     plt.figure(figsize=(10,8))
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    np.random.seed(0)
    y_test = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test.shape[1]
    for clf, clf_name in zip(all_clfs, clf_names):
        #two classes 
        if n_classes <= 2:
            probs = clf.predict_proba(X_test)
            preds = probs[:,1]
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, preds)
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            roc_scores[clf_name] = roc_auc
            plt.plot(fpr, tpr, 'b', label = '%s (AUC = %0.3f)' % (clf_name,roc_auc), c=np.random.rand(3,))
            plt.legend(loc = 'lower right')         
        #multi classes
        else:
            y_score = clf.predict_proba(X_test)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:,i],y_score[:,i])
                roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(),y_score.ravel())
            roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
            roc_scores[clf_name] = roc_auc["micro"]
            plt.plot(fpr["micro"], tpr["micro"], 'b', label = '%s (Micro AUC = %0.3f)' % 
                     (clf_name,roc_auc["micro"]), c=np.random.rand(3,))
            plt.legend(loc = 'lower right')        
    return roc_scores

#plot tools
def plot_roc_curve2(X_test, y_test, all_clfs, clf_names):
    roc_scores = dict()
#     plt.figure(figsize=(10,8))
    plt.title('Receiver Operating Characteristic')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    np.random.seed(0)

    for clf, clf_name, X, y in zip(all_clfs, clf_names, X_test, y_test):
        y = label_binarize(y, classes=np.unique(y))
        n_classes = y.shape[1]        
        #two classes 
        if n_classes <= 2:
            probs = clf.predict_proba(X)
            preds = probs[:,1]
            fpr, tpr, _ = sklearn.metrics.roc_curve(y, preds)
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            roc_scores[clf_name] = roc_auc
            plt.plot(fpr, tpr, 'b', label = '%s (AUC = %0.3f)' % (clf_name,roc_auc), c=np.random.rand(3,))
            plt.legend(loc = 'lower right')         
        #multi classes
        else:
            y_score = clf.predict_proba(X)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y[:,i],y_score[:,i])
                roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y.ravel(),y_score.ravel())
            roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
            roc_scores[clf_name] = roc_auc["micro"]
            plt.plot(fpr["micro"], tpr["micro"], 'b', label = '%s (Micro AUC = %0.3f)' % 
                     (clf_name,roc_auc["micro"]), c=np.random.rand(3,))
            plt.legend(loc = 'lower right')        
    return roc_scores

def plot_pr_curve(X_test, y_test, all_clfs, clf_names):
    if type(X_test) is list:
        return plot_pr_curve2(X_test, y_test, all_clfs, clf_names)    
    pr_scores = dict()
#     plt.figure(figsize=(10,8))
    plt.title('Precision-Recall Curve')
    plt.plot([0, 1], [1, 0],'r--')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    np.random.seed(0)
    y_test = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test.shape[1]
    
    for clf, clf_name in zip(all_clfs, clf_names):
        #two classes
        if n_classes <= 2:
            probs = clf.predict_proba(X_test)
            preds = probs[:,1]
            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, preds)
            pr_auc = sklearn.metrics.auc(recall, precision)
            pr_scores[clf_name] = pr_auc
            plt.plot(recall, precision, 'b', label = '%s (AUC = %0.3f)' % (clf_name,pr_auc), c=np.random.rand(3,))
            plt.legend(loc = 'lower left')
        #multi classes
        else:
            y_score = clf.predict_proba(X_test)
            # Compute ROC curve and ROC area for each class
            precision = dict()
            recall = dict()
            pr_auc = dict()
            for i in range(n_classes):
                precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_test[:,i],y_score[:,i])
                pr_auc[i] = sklearn.metrics.auc(recall[i], precision[i])
            # Compute micro-average ROC curve and ROC area
            precision["micro"], recall["micro"], _ = sklearn.metrics.precision_recall_curve(y_test.ravel(),y_score.ravel())
            pr_auc["micro"] = sklearn.metrics.auc(recall["micro"], precision["micro"])  
            pr_scores[clf_name] = pr_auc["micro"]
            plt.plot(recall["micro"], precision["micro"], 'b', label = '%s (Micro AUC = %0.3f)' % 
                     (clf_name,pr_auc["micro"]), c=np.random.rand(3,))
            plt.legend(loc = 'lower left')
    return pr_scores

def plot_pr_curve2(X_test, y_test, all_clfs, clf_names):
    pr_scores = dict()
#     plt.figure(figsize=(10,8))
    plt.title('Precision-Recall Curve')
    plt.plot([0, 1], [1, 0],'r--')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    np.random.seed(0)
    
    for clf, clf_name, X, y in zip(all_clfs, clf_names, X_test, y_test):
        y = label_binarize(y, classes=np.unique(y))
        n_classes = y.shape[1]        
        #two classes
        if n_classes <= 2:
            probs = clf.predict_proba(X)
            preds = probs[:,1]
            precision, recall, _ = sklearn.metrics.precision_recall_curve(y, preds)
            pr_auc = sklearn.metrics.auc(recall, precision)
            pr_scores[clf_name] = pr_auc
            plt.plot(recall, precision, 'b', label = '%s (AUC = %0.3f)' % (clf_name,pr_auc), c=np.random.rand(3,))
            plt.legend(loc = 'lower left')
        #multi classes
        else:
            y_score = clf.predict_proba(X)
            # Compute ROC curve and ROC area for each class
            precision = dict()
            recall = dict()
            pr_auc = dict()
            for i in range(n_classes):
                precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y[:,i],y_score[:,i])
                pr_auc[i] = sklearn.metrics.auc(recall[i], precision[i])
            # Compute micro-average ROC curve and ROC area
            precision["micro"], recall["micro"], _ = sklearn.metrics.precision_recall_curve(y.ravel(),y_score.ravel())
            pr_auc["micro"] = sklearn.metrics.auc(recall["micro"], precision["micro"])  
            pr_scores[clf_name] = pr_auc["micro"]
            plt.plot(recall["micro"], precision["micro"], 'b', label = '%s (Micro AUC = %0.3f)' % 
                     (clf_name,pr_auc["micro"]), c=np.random.rand(3,))
            plt.legend(loc = 'lower left')
    return pr_scores

#plot non-normalized confusion matrix(default)
def plot_confusion_matrix(estimator, X_test, y_test, classes, cmap=plt.cm.GnBu, title='Confusion Matrix', threshold=None, normalize=None):
    import itertools
    from sklearn.metrics import confusion_matrix
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    if threshold == None:
        y_pred = estimator.predict(X_test)
    else:
        y_proba = estimator.predict_proba(X_test)
        y_pred = y_proba[:,1] > threshold
    
    cm = confusion_matrix(y_test, y_pred)
    
    np.set_printoptions(precision=2)
    recall = cm[1,1]/(cm[1,0]+cm[1,1])
    acc = (cm[0,0]+cm[1,1])/(cm.sum())
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation=90)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.title(title+' (Recall: %.2f%%  Acc: %.2f%%)'%(recall*100,acc*100))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)    
    plt.colorbar(im, cax=cax)

#plot confusion matrix with given thresholds    
def plot_cm_thresholds(estimator, X_test, y_test, classes, thresholds):
    plt.figure(figsize=(20,15))
    for j,i in enumerate(thresholds):
        plt.subplot(3,4,j+1)
        plot_confusion_matrix(estimator, X_test, y_test, classes, title='Threshold >= %s'%i, threshold=i)
    plt.show()
        

def plot_embedding(X,y,classes,title,output_path=None):
    comp1 = pd.DataFrame(X).to_numpy()[:,0]
    comp2 = pd.DataFrame(X).to_numpy()[:,1]
    y = pd.DataFrame(y).to_numpy().ravel()
    
    plt.figure(figsize=(15,10))
    color_map = plt.cm.get_cmap('tab10')
    
#     #plot without labels (faster)
#     plt.scatter(comp1,comp2,c=y,cmap=color_map,alpha=0.5)

    #plot labels
    labels = np.array(classes)[y]
    class_num = set()
    for x1,x2,c,l in zip(comp1,comp2,color_map(y),labels):
        if len(class_num)==10:
            break
        plt.scatter(x1,x2,c=[c],label=l,alpha=1)
        class_num.add(l)
        
    #remvoe duplicate labels    
    hand, labl = plt.gca().get_legend_handles_labels()
    handout=[]
    lablout=[]
    for h,l in zip(hand,labl):
        if l not in lablout:
            lablout.append(l)
            handout.append(h)
    plt.title(title)
    plt.xlabel('Component One')
    plt.ylabel('Component Two')
    plt.legend(handout, lablout,fontsize=20)
    if output_path:
        plt.savefig(output_path+title)
    plt.show()
    
    
#plot estimator score and x_train dimensions/features
def plot_dimension_accuracy_curve(x_train, y_train, x_test, y_test, estimators, output_path=None):
    D = x_train.shape[1]
    est_scores = dict()
    for estimator in estimators:
        try:
            est_name = str(estimator).split('(')[0]
            train_scores = []
            test_scores = []
            for d in range(1,D+1):
                xtrain, xtest = dimension_reduction(x_train, x_test, n_components=d, verbose=False)
                est = estimator.fit(xtrain, y_train)
                train_scores.append(est.score(xtrain, y_train))
                test_scores.append(est.score(xtest, y_test))
            est_scores[est_name] = (train_scores, test_scores)
        except:
            pass
        
    color_map = plt.cm.get_cmap('tab10')
    colors = color_map(np.arange(len(est_scores)))
    plt.figure(figsize=(10,6))
    plt.title('Dimension - Accuracy Curves')
    plt.xlim(1,D)
    plt.xlabel('Dimension of Features')
    plt.ylabel('Accuracy')
    handles = []
    for est,color in zip(est_scores,colors):
        l1, = plt.plot(np.arange(1,D+1), est_scores[est][0], color=color )
        l2, = plt.plot(np.arange(1,D+1), est_scores[est][1], '--', color=color)
        handles.append([l1,l2])
    legend1 = plt.legend(handles[0], ["Train", "Test"], loc=2)
    plt.legend([h[0] for h in handles], est_scores.keys(), loc=4)
    plt.gca().add_artist(legend1)
    if output_path:
        plt.savefig(output_path+'dimension_acc_curves')
    plt.show()
    
#plot estimator score and x_train dimensions/features
def plot_dimension_roc_curve(x_train, y_train, x_test, y_test, estimators, output_path=None):
    D = x_train.shape[1]
    est_scores = dict()
    for estimator in estimators:
        try:
            est_name = str(estimator).split('(')[0]
            train_scores = []
            test_scores = []
            for d in range(1,D+1):
                xtrain, xtest = dimension_reduction(x_train, x_test, n_components=d, verbose=False)
                est = estimator.fit(xtrain, y_train)
                probs = est.predict_proba(xtrain)
                preds = probs[:,1]
                fpr, tpr, _ = sklearn.metrics.roc_curve(y_train, preds)
                roc_auc = sklearn.metrics.auc(fpr, tpr)
                train_scores.append(roc_auc)
                
                probs = est.predict_proba(xtest)
                preds = probs[:,1]
                fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, preds)
                roc_auc = sklearn.metrics.auc(fpr, tpr)
                test_scores.append(roc_auc)
            est_scores[est_name] = (train_scores, test_scores)
        except:
            pass
        
    color_map = plt.cm.get_cmap('tab10')
    colors = color_map(np.arange(len(est_scores)))
    plt.figure(figsize=(10,6))
    plt.title('Dimension - ROC_AUC Curves')
    plt.xlim(1,D)
    plt.xlabel('Dimension of Features')
    plt.ylabel('ROC_AUC')
    handles = []
    for est,color in zip(est_scores,colors):
        l1, = plt.plot(np.arange(1,D+1), est_scores[est][0], color=color )
        l2, = plt.plot(np.arange(1,D+1), est_scores[est][1], '--', color=color)
        handles.append([l1,l2])
    legend1 = plt.legend(handles[0], ["Train", "Test"], loc=2)
    plt.legend([h[0] for h in handles], est_scores.keys(), loc=4)
    plt.gca().add_artist(legend1)
    if output_path:
        plt.savefig(output_path+'dimension_roc_curves')
    plt.show()

#plot estimator score and x_train dimensions/features
def plot_dimension_pr_curve(x_train, y_train, x_test, y_test, estimators, output_path=None):
    D = x_train.shape[1]
    est_scores = dict()
    for estimator in estimators:
        try:
            est_name = str(estimator).split('(')[0]
            train_scores = []
            test_scores = []
            for d in range(1,D+1):
                xtrain, xtest = dimension_reduction(x_train, x_test, n_components=d, verbose=False)
                est = estimator.fit(xtrain, y_train)
                probs = est.predict_proba(xtrain)
                preds = probs[:,1]
                precision, recall, _ = sklearn.metrics.precision_recall_curve(y_train, preds)                
                pr_auc = sklearn.metrics.auc(recall, precision)
                train_scores.append(pr_auc)
                
                probs = est.predict_proba(xtest)
                preds = probs[:,1]
                precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, preds)                
                pr_auc = sklearn.metrics.auc(recall, precision)
                test_scores.append(pr_auc)
            est_scores[est_name] = (train_scores, test_scores)
        except:
            pass
        
    color_map = plt.cm.get_cmap('tab10')
    colors = color_map(np.arange(len(est_scores)))
    plt.figure(figsize=(10,6))
    plt.title('Dimension - PR_AUC Curves')
    plt.xlim(1,D)
    plt.xlabel('Dimension of Features')
    plt.ylabel('PR_AUC')
    handles = []
    for est,color in zip(est_scores,colors):
        l1, = plt.plot(np.arange(1,D+1), est_scores[est][0], color=color )
        l2, = plt.plot(np.arange(1,D+1), est_scores[est][1], '--', color=color)
        handles.append([l1,l2])
    legend1 = plt.legend(handles[0], ["Train", "Test"], loc=2)
    plt.legend([h[0] for h in handles], est_scores.keys(), loc=4)
    plt.gca().add_artist(legend1)
    if output_path:
        plt.savefig(output_path+'dimension_pr_curves')
    plt.show()    