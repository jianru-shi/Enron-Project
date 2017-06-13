
"""
This module provides pipeline and parameters creation functions in building
POI prediction models
Available functions include:
- accuracy_check: Scoring function to be used in comparing selected models
- NB_pipeline: Make a pipeline for cross-validated grid search for the
    Naive Bayes Gaussian Model.
    
- NB_params: Make a parameters dictionary for cross-validated
    grid search for the Naive Bayes Gaussian Model.
    
- Tree_pipeline: Make a pipeline for cross-validated grid search for the
    Decision Tree Classifier.
    
- Tree_params: Make a parameters dictionary for cross-validated grid search
    for the Decision Tree Classifier.
        
- SVC_pipeline: Make a pipeline for cross-validated grid search for the
    Support Vector Machines Classifier.
    
- SVC_params: Make a parameters dictionary for cross-validated grid search
    for the Support Vector Machines Classifier.
        
"""

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn import preprocessing


def accuracy_check (predicted, actural): 
    """
    Args:
        predicted: a list of predicted labels 
        actural: a list of labels of test dataset
    Returns:
        Accuracy, Precision, Recall, f1, and f2 scores
    """
    TP=sum(predicted+actural==2)
    TN=sum(predicted+actural==0)
    FP=sum(predicted)-TP
    FN=sum(actural)-TP
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1 = 2.0 * TP/(2*TP + FP+FN)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    return accuracy, precision, recall, f1, f2


## Naive Bayes 


def NB_pipeline():
    '''Make a pipeline for cross-validated grid search for the
        Naive Bayes Classifier.
    
    This function makes a pipeline which:
        1. Selects the KBest features using Anova F-value scoring for
            classification.
        2. Uses KBest features in Naive Bayes Classifier.
    '''
    
    NB_clf = Pipeline([
      ('selection', SelectKBest(f_classif)),
      ('classification', GaussianNB())
    ])
    
    return  NB_clf


def NB_Params():
    '''Make a parameters dictionary for cross-validated grid search for the
       Naive Bayes Classifier.
    
    This function makes a parameter dictionary to search over.
    
    Parameters searched over include:
        SelectKBest:        
            1. k: Number of KBest features to select.

    Returns:
        A dictionary of parameters to pass into an sk-learn grid-search 
            pipeline. 
    '''

    NB_Params={'selection__k': [5,6,7,9,11,13,15,17]
                }
    return NB_Params




## DT
def Tree_pipeline():
    '''Make a pipeline for cross-validated grid search for the
        Decision Tree Classifier.
    
    This function makes a pipeline which:
        1. Selects the KBest features using Anova F-value scoring for
            classification.
        2. Uses KBest features in Decision Tree Classifier.
    
    '''
    Tree_clf = Pipeline([
      ('selection', SelectKBest(f_classif)),
      ('classification', tree.DecisionTreeClassifier(random_state=42))
    ])
    
    return Tree_clf


def Tree_Params():
    '''Make a parameters dictionary for cross-validated grid search for the
       Decision Tree Classifier.
    
    This function makes a parameter dictionary to search over.
    
    Parameters searched over include:
        SelectKBest:        
            1. k: Number of KBest features to select.
        DecisionTreeClassifier:
            1. criterion: The function to measure the quality of a split.
            2. min_samples_split: The minimum number of samples required 
               to split an internal node
            3. max_features: The number of features to consider when 
               looking for the best split
    Returns:
        A dictionary of parameters to pass into an sk-learn grid-search 
            pipeline. 
    '''

    Tree_Params={'selection__k': [5,7,9,11,13,15,17],
                'classification__criterion': ['gini', 'entropy'],
                'classification__min_samples_split':[2,4,6],
                'classification__max_features':['log2','auto']
                }
    return Tree_Params



## SVC


def SVC_pipeline():
    '''Make a pipeline for cross-validated grid search for the
        Support Vector Machines Classifier.
    
    This function makes a pipeline which:
        1. Scales the features between 0-1
        2. Selects the KBest features using Anova F-value scoring for
            classification.
        3. Uses KBest features in Support Vector Machines Classifier.
    
    '''

    SVC_clf=Pipeline([
        ('scaler', preprocessing.MinMaxScaler()),
        ('selection', SelectKBest(f_classif)),
        ('classification', SVC(random_state=42) )
    ])
    
    return SVC_clf 

def SVC_Params():
    '''Make a parameters dictionary for cross-validated grid search for the
        Support Vector Machines Classifier.
    
    This function makes a parameter dictionary to search over.
    
    Parameters searched over include:
        SelectKBest:        
            1. k: Number of KBest features to select.
        SVC:
            1. C: Value of the regularization constraint.
            2. kernel: Specifies the kernel type to be used in the algorithm
            3. gamma: Kernel coefficient for 'rbf' kernel
    Returns:
        A dictionary of parameters to pass into an sk-learn grid-search 
            pipeline. 
    '''
    SVC_params = {'selection__k': [7, 9, 11,13,15, 17, 21],
              'classification__C': [1e-5, 1e-2, 1e-1, 1, 10, 100],
              'classification__kernel':["linear", "rbf","poly"],
              'classification__gamma':[0,10,100]
              }
    return SVC_params














