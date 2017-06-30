from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
train = train.replace({'DetectedCamera':mapping})
test = test.replace({'DetectedCamera':mapping})
mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
train = train.replace({'SignFacing (Target)':mapping})

labels_train = train['SignFacing (Target)']
test_id = test['Id']
#drop columns
train.drop(['SignFacing (Target)','Id'], inplace=True, axis=1)
test.drop('Id',inplace=True,axis=1)

# param_dist = {"max_depth": [3, None],
#               "max_features": sp_randint(1, 11),
#               "min_samples_split": sp_randint(1, 11),
#               "min_samples_leaf": sp_randint(1, 11),
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}

# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 5],
#               "min_samples_split": [5, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["gini", "entropy"]}

# previous best score - 99.90094 for mapping = {'Front':0, 'Left':1, 'Rear':2, 'Right':3}
clf1 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =15, max_features = 4, min_samples_leaf = 8,min_samples_split=5)
clf2 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =70, max_features = "auto", min_samples_leaf = 30)
clf3 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =35, max_features = 4, min_samples_leaf = 17)
#clf4 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =35, max_features = "auto", min_samples_leaf = 15)
#clf5 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state = 0, max_features = "auto", min_samples_leaf = 5)

eclf = VotingClassifier(estimators=[('rf', clf1), ('rf', clf2), ('rf', clf3)], voting='soft')


#clf = svm.SVC()
# clf1 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =15, max_features = "auto", min_samples_leaf = 5)
# clf2 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =70, max_features = "auto", min_samples_leaf = 30)
# clf3 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =35, max_features = "auto", min_samples_leaf = 15)
# clf4 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =5, max_features = "auto", min_samples_leaf = 5)
# clf5 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state = 0, max_features = "auto", min_samples_leaf = 5)
# clf5 = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1,random_state =68, max_features = "auto", min_samples_leaf = 29)
#clf4 = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=1, random_state=0)
#clf = RandomForestClassifier(n_estimators = 1000, oob_score = True, n_jobs = -1, max_features = 0.5, min_samples_leaf = 30,random_state =70)
# grid_search = GridSearchCV(clf, param_grid=param_grid)
# grid_search.fit(train, labels_train)
#eclf = VotingClassifier(estimators=[('rf', clf1), ('rf', clf2), ('rf', clf3), ('rf', clf4), ('rf', clf5)], voting='soft')
# clf = RandomForestClassifier(criterion='gini', 
#                              n_estimators=500,
#                              min_samples_split=5,
#                              min_samples_leaf=3,
#                              max_features='auto',
#                              oob_score=True,
#                              random_state=10,
#                              n_jobs=-1)

eclf.fit(train,labels_train)
pred = eclf.predict_proba(test)
columns = ['Front','Left','Rear','Right']
sub = pd.DataFrame(data=pred, columns=columns)
sub['Id'] = test_id
sub = sub[['Id','Front','Left','Rear','Right']]
sub.to_csv('india_hacks_here_maps_result.csv', index=False, float_format='%0.6f')
