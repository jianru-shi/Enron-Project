#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline


from tester import dump_classifier_and_data

from tester import *


####################################################################################
### Task 1: Select what features you'll use.                                     ###
### features_list is a list of strings, each of which is a feature name.         ###
### The first feature must be "poi".                                             ###
### The initial feature_list contians all features, final feature_list will      ###
### be updated after feature selection                                           ###
####################################################################################
features_list = ['poi','salary', 'deferral_payments',\
                 'total_payments', 'loan_advances', 'bonus',\
                 'restricted_stock_deferred', 'deferred_income',\
                 'total_stock_value', 'expenses', 'exercised_stock_options',\
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',\
                'to_messages', 'from_poi_to_this_person',\
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


############################# Data Overview ##################################
    
print 'The number of observations:',len(data_dict), '\n' \
'The number of features:',  len(features_list)

# convert the dictionary to pandas dataframe for data wrangling and cleaning

data_df=pd.DataFrame.from_dict(data_dict,orient='index', dtype='float')
data_df[['poi']]=data_df[['poi']].astype('bool')
data_df=data_df.replace('NaN', np.nan) 

# missing values
print 'Summary of missing values:'
data_df.isnull().sum()

# count labels frequencty

print 'Counts of classies:'
data_df['poi'].value_counts()


############################# Data Cleaning ####################################
### Task 2: Remove outliers

## Visualize outliers in box-plot
data_df[['salary', 'deferral_payments','total_payments',\
         'bonus','total_stock_value', 'exercised_stock_options',]].plot.box()

## identify observations that contains outlier 
data_df[['total_stock_value', 'total_payments']].describe()
data_df[data_df['total_stock_value']>200000000]

# Remove the row with errors
data_df.drop(['TOTAL'], inplace=True)

# Check the abnormal observations has negative total stock value
data_df[data_df['total_stock_value']<0]

# plot the cleaned data 
data_df[['salary', 'deferral_payments','total_payments',\
         'bonus','total_stock_value', 'exercised_stock_options',\
        'long_term_incentive', 'restricted_stock', 'director_fees',]].plot.box()

# Check the outlier in total payments 
# Keep the oberservation since Lay_Kennethl is a person name
data_df[data_df['total_payments']>100000000]


### Task 3: Create new feature(s)

data_df['to_poi_fraction']=data_df['from_this_person_to_poi']/data_df['from_messages']
data_df['from_poi_fraction']=data_df['from_poi_to_this_person']/data_df['to_messages']

### filled missing values with 0 
data_df=data_df.fillna(0)

### update the features_list by adding two new features
features_list = ['poi','salary', 'deferral_payments',
                 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi',
                'to_poi_fraction','from_poi_fraction'] 


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
from feature_format import featureFormat, targetFeatureSplit
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Split training dataset and test dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
########################################################################    
### Task 4: Try a varity of classifiers                              ###
### Three classifiers were tested:                                   ###
### a. Naive Bayes Gaussian                                          ###
### b. Decision Tree                                                 ###
### c. SVC                                                           ###
### Best parameters for each classifier were found via GridSearchcv  ###
########################################################################

### Import models and parameters from select_models.py
from select_models import *

### Classifier_1 Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

clf=NB_pipeline()
Params=NB_Params() 
grid_searcher = GridSearchCV(clf, param_grid=Params, cv=10, scoring='f1')

grid_searcher.fit(features_train, labels_train)
print 'The f1 score from Naive Bayes model is:', grid_searcher.best_score_
result_df=pd.DataFrame.from_dict(grid_searcher.cv_results_)
pd.DataFrame.from_dict(grid_searcher.best_params_, orient='index') 

### Final Naive Bayes Classifier
NB_clf=Pipeline([
    ('selection', SelectKBest(f_classif, k=13)),
    ('classification', GaussianNB())
])


### Classifier_2 Decision Tree
from sklearn import tree

clf=Tree_pipeline()
Params=Tree_Params() 
grid_searcher = GridSearchCV(clf, param_grid=Params, cv=10, scoring='f1')

grid_searcher.fit(features_train, labels_train)
print 'The f1 score from Decision Tree model is:', grid_searcher.best_score_
result_df=pd.DataFrame.from_dict(grid_searcher.cv_results_)
pd.DataFrame.from_dict(grid_searcher.best_params_, orient='index') 

### Final tree model
Tree_clf = Pipeline([
  ('selection', SelectKBest(f_classif, k=7)),
  ('classification', tree.DecisionTreeClassifier(criterion='entropy',\
                                                 min_samples_split=4,\
                                                 random_state=42,\
                                                 max_features='log2'))
])

### Classifier_3 SVC 
from sklearn.svm import SVC
from sklearn import preprocessing

clf=SVC_pipeline()
Params=SVC_Params()
grid_searcher = GridSearchCV(clf, param_grid=Params, cv=10, scoring='f1')

grid_searcher.fit(features_train, labels_train)
print 'The f1 score for SVC classifier is:', grid_searcher.best_score_

result_df=pd.DataFrame.from_dict(grid_searcher.cv_results_)
pd.DataFrame.from_dict(grid_searcher.best_params_, orient='index') 

### Final SVC model
SVC_clf=Pipeline([
    ('scaler', preprocessing.MinMaxScaler()),
    ('selection', SelectKBest(f_classif, k=7)),
    ('classification', SVC(kernel='poly', gamma=100, C=10, random_state=42) )
])


###################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall ###
### using our testing script. Check the tester.py script in the final project   ###
### folder for details on the evaluation method, especially the test_classifier ###
### function. Because of the small size of the dataset, the script uses         ###
### stratified shuffle split cross validation.                                  ###
###################################################################################


### Compare three models using train and test data.
### Returns Accuracy, Precision, Recall, and f1 scores
### Results may be different with the results from tester_classifiers since
### test_classifier using cross validation.

NB_clf.fit(features_train,labels_train)
Tree_clf.fit(features_train,labels_train)
SVC_clf.fit(features_train,labels_train)

classifers_list=[NB_clf,Tree_clf,SVC_clf]
classifer_name_list=['Gaussian Naive Bayes','Decision Tree', 'SVC']

clf_scores={}
for ii in range(0, len(classifers_list)):
    classifier=classifers_list[ii]
    classifer_name=classifer_name_list[ii]
    pred=classifier.predict(features_test)
    scores=accuracy_check(pred, labels_test)
    scores_dict={'Accuracy': scores[0], 'Precision': scores[1],\
     'Recall':scores[2], 'f1':scores[3], 'f2':scores[4]}
    clf_scores[classifer_name]=scores_dict

pd.DataFrame.from_dict(clf_scores, orient='index')


###################################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can      ###
### check your results. You do not need to change anything below, but make sure ###
### that the version of poi_id.py that you submit can be run on its own and     ###
### generates the necessary .pkl files for validating your results.             ###
###################################################################################

clf=Tree_clf

full_features_list = ['poi','salary', 'deferral_payments',
                 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi',
                'to_poi_fraction','from_poi_fraction'] 

support=clf.named_steps['selection'].get_support()
features_list=list(np.array(full_features_list[1:])[support])
features_list=['poi']+features_list

dump_classifier_and_data(clf, my_dataset, features_list)

### Plot features scores
scores=clf.named_steps['selection'].scores_
scores_dict={}
for i in range (0, len(scores)):
    feature_name=full_features_list[i+1]
    score=scores[i]
    scores_dict[feature_name]=score
scores_df=pd.DataFrame.from_dict(scores_dict, orient='index')
scores_df.columns=['Score']
scores_df.sort_values(['Score'], ascending=False).plot(kind='bar')

### Features importances
scores=clf.named_steps['classification'].feature_importances_
scores_dict={}
for i in range (0, len(scores)):
    feature_name=features_list[i+1]
    score=scores[i]
    scores_dict[feature_name]=score
scores_df=pd.DataFrame.from_dict(scores_dict, orient='index')
scores_df.columns=['Score']
scores_df.sort_values(['Score'], ascending=False).plot(kind='bar')


