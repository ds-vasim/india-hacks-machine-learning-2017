from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np
import pandas as pd
import json
import re

file_name_1 = "train_data.json"
with open(file_name_1, 'r') as jsonfile1:
    data_dict_1 = json.load(jsonfile1)
    
file_name_2 = "test_data.json"
with open(file_name_2, 'r') as jsonfile2:
    data_dict_2 = json.load(jsonfile2)

train = pd.DataFrame.from_dict(data_dict_1, orient='index')
train.reset_index(level=0, inplace=True)
train.rename(columns = {'index':'ID'},inplace=True)

test = pd.DataFrame.from_dict(data_dict_2, orient='index')
test.reset_index(level=0, inplace=True)
test.rename(columns = {'index':'ID'},inplace=True)

train = train.replace({'segment':{'pos':1,'neg':0}})

train['g1'] = [re.sub(pattern='\:\d+',repl='',string=x) for x in train['genres']]
train['g1'] = train['g1'].apply(lambda x: x.split(','))

train['g2'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in train['dow']]
train['g2'] = train['g2'].apply(lambda x: x.split(','))

t1 = pd.Series(train['g1']).apply(frozenset).to_frame(name='t_genre')
t2 = pd.Series(train['g2']).apply(frozenset).to_frame(name='t_dow')

for t_genre in frozenset.union(*t1.t_genre):
    t1[t_genre] = t1.apply(lambda _: int(t_genre in _.t_genre), axis=1)

for t_dow in frozenset.union(*t2.t_dow):
    t2[t_dow] = t2.apply(lambda _: int(t_dow in _.t_dow), axis = 1)

train = pd.concat([train.reset_index(drop=True), t1], axis=1)
train = pd.concat([train.reset_index(drop=True), t2], axis=1)

# for test data

test['g1'] = [re.sub(pattern='\:\d+',repl='',string=x) for x in test['genres']]
test['g1'] = test['g1'].apply(lambda x: x.split(','))

test['g2'] = [re.sub(pattern='\:\d+', repl='', string = x) for x in test['dow']]
test['g2'] = test['g2'].apply(lambda x: x.split(','))

t1_te = pd.Series(test['g1']).apply(frozenset).to_frame(name='t_genre')
t2_te = pd.Series(test['g2']).apply(frozenset).to_frame(name='t_dow')

for t_genre in frozenset.union(*t1_te.t_genre):
    t1_te[t_genre] = t1_te.apply(lambda _: int(t_genre in _.t_genre), axis=1)

for t_dow in frozenset.union(*t2_te.t_dow):
    t2_te[t_dow] = t2_te.apply(lambda _: int(t_dow in _.t_dow), axis = 1)

test = pd.concat([test.reset_index(drop=True), t1_te], axis=1)
test = pd.concat([test.reset_index(drop=True), t2_te], axis=1)

#the rows aren't list exactly. They are object, so we convert them to list and extract the watch time
w1 = train['titles']
w1 = w1.str.split(',')

#create a nested list of numbers
main = []
for i in np.arange(train.shape[0]):
    d1 = w1[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main.append(nest)

blanks = []
for i in np.arange(len(main)):
    if '' in main[i]:
        #print ("{} blanks found".format(len(blanks)))
        blanks.append(i)
        
#replacing blanks with 0
for i in blanks:
    main[i] = [x.replace('','0') for x in main[i]]
    
#converting string to integers
main = [[int(y) for y in x] for x in main]

#adding the watch time
tosum = []
for i in np.arange(len(main)):
    s = sum(main[i])
    tosum.append(s)

train['title_sum'] = tosum

#making changes in test data
w1_te = test['titles']
w1_te = w1_te.str.split(',')

main_te = []
for i in np.arange(test.shape[0]):
    d1 = w1_te[i]
    nest = []
    nest = [re.sub(pattern = '.*\:', repl=' ', string= d1[k]) for k in list(np.arange(len(d1)))]
    main_te.append(nest)

blanks_te = []
for i in np.arange(len(main_te)):
    if '' in main_te[i]:
        #print ("{} blanks found".format(len(blanks_te)))
        blanks_te.append(i)
        
#replacing blanks with 0
for i in blanks_te:
    main_te[i] = [x.replace('','0') for x in main_te[i]]
    
#converting string to integers
main_te = [[int(y) for y in x] for x in main_te]

#adding the watch time
tosum_te = []
for i in np.arange(len(main_te)):
    s = sum(main_te[i])
    tosum_te.append(s)

test['title_sum'] = tosum_te

#count variables
def wcount(p):
    return p.count(',')+1

train['title_count'] = train['titles'].map(wcount)
train['genres_count'] = train['genres'].map(wcount)
train['cities_count'] = train['cities'].map(wcount)
train['tod_count'] = train['tod'].map(wcount)
train['dow_count'] = train['dow'].map(wcount)


test['title_count'] = test['titles'].map(wcount)
test['genres_count'] = test['genres'].map(wcount)
test['cities_count'] = test['cities'].map(wcount)
test['tod_count'] = test['tod'].map(wcount)
test['dow_count'] = test['dow'].map(wcount)

test_id = test['ID']
train.drop(['ID','cities','dow','genres','titles','tod','g1','g2','t_genre','t_dow'], inplace=True, axis=1)
test.drop(['ID','cities','dow','genres','titles','tod','g1','g2','t_genre','t_dow'], inplace=True, axis=1)

target_label = train['segment']


#drop unnecessary columns
train.drop('segment',inplace=True,axis=1)

#Best result - 0.79510
clf = RandomForestClassifier(n_estimators=500,max_depth=12, max_features=10,n_jobs = -1,random_state =10,min_samples_leaf = 40)

#clf1 = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0,max_depth=12, random_state=10, min_samples_leaf = 40)
#clf2 = AdaBoostClassifier(n_estimators = 1000, learning_rate=0.5, random_state=10)
#clf = RandomForestClassifier(n_estimators=100,max_depth=4, max_features=5,n_jobs = -1,random_state =10,min_samples_leaf = 40)
# clf1 = RandomForestClassifier(n_estimators=500,max_depth=12, max_features=10,n_jobs = -1,random_state =10,min_samples_leaf = 40)
# clf2 = RandomForestClassifier(n_estimators=500,max_depth=12, max_features=8,n_jobs = -1,random_state =5,min_samples_leaf = 5)
# clf3 = RandomForestClassifier(n_estimators=500,max_depth=12, max_features=6,n_jobs = -1,random_state =20,min_samples_leaf = 20)
#clf2 = GradientBoostingClassifier(n_estimators=500,max_depth=12, max_features=10,random_state =10,min_samples_leaf = 40)
#clf3 = AdaBoostClassifier(n_estimators=500,learning_rate=1.0, random_state=10)
#eclf = VotingClassifier(estimators=[('rf', clf1), ('rf', clf2), ('rf', clf3)], voting='soft')

clf.fit(train, target_label)
pred = clf.predict_proba(test)

columns = ['segment']
sub = pd.DataFrame(data=pred[:,1], columns=columns)
sub['ID'] = test_id
sub = sub[['ID','segment']]
sub.to_csv('india_hacks_hotstar_result.csv', index=False)
