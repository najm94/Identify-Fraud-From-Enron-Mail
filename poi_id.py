#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import grid_search 
from sklearn import tree
from sklearn import preprocessing

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','deferral_payments', 
                 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 
                 'director_fees','to_messages','from_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)
#Creating two new features 'fraction_to_POI' and 'fraction_from_POI'
for key in data_dict:
    if (data_dict[key]['from_this_person_to_poi']=='NaN') or (data_dict[key]['from_messages']=='NaN'):
      data_dict[key]['fraction_to_POI'] = 0
    else:  
      data_dict[key]['fraction_to_POI'] = (1.0*data_dict[key]['from_this_person_to_poi']/data_dict[key]['from_messages'])   
    if (data_dict[key]['from_poi_to_this_person']=='NaN') or (data_dict[key]['to_messages']=='NaN'):
      data_dict[key]['fraction_from_POI'] = 0
    else:  
      data_dict[key]['fraction_from_POI'] = (1.0*data_dict[key]['from_poi_to_this_person']/data_dict[key]['to_messages'])
### Store to my_dataset for easy export below.
my_dataset = data_dict

#Plotting new feature
features_list = ["poi", "fraction_from_POI", "fraction_to_POI"]
data = featureFormat(my_dataset, features_list)

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="y", marker="*")
plt.xlabel("Fraction of emails from poi")
plt.ylabel("Fraction of emails to poi")
plt.show()





### Extract features and labels from dataset for local testing
features_list = ['poi','salary','fraction_to_POI','bonus','fraction_from_POI','deferral_payments', 
                 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 
                 'director_fees','to_messages','from_messages'] 

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#selectKbest to find the most important features
selector = SelectKBest(f_classif, k=5)
selector.fit(features,labels)
selector.scores_
#Selecting most important 5 features
features_list=['poi','salary','fraction_to_POI','exercised_stock_options','total_stock_value','bonus']
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Kfold for split and validate algorithm
#Using KFold validation
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#Before tuning
#Decision tree
clf = DecisionTreeClassifier(min_samples_split=4,min_samples_leaf=4)
clf.fit(features_train,labels_train)

#KNearest Neighbor
clf = KNeighborsClassifier()
clf.fit(features_test,labels_test)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#Finding out the best parameters for the decision tree classifier
parameters = {'min_samples_leaf':[2,3,4,5], 'min_samples_split':[2,3,4,5]}
my_clf = tree.DecisionTreeClassifier()
clf = grid_search.GridSearchCV(my_clf, parameters)
clf = clf.fit(features, labels)
clf.best_estimator_
#The parameters given by GridSearchCV does not give good precision and recall scores


#Feature scaling using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

#Choosing KNeighbors classifier as the algorithm since for n_neighbors=3
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(features_test,labels_test)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


dump_classifier_and_data(clf, my_dataset, features_list)
    
