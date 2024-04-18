# importing Libraries
import numpy as np # For numerical analysis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting graphs
import nltk # for text processing
import os # for system based operations
import seaborn as sns
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn import decomposition, ensemble
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# Lets first get the Data in a Dataframe
spam_test_dataframe = pd.read_csv(os.getcwd() + '/sms_spam.csv',names= ['label', 'feature']) # read csv as we have a csv file to read

spam_test_dataframe.head() # to print first 5 rows of data frame


spam_test_dataframe.dropna()
spam_test_dataframe=spam_test_dataframe.iloc[1:]

spam_test_dataframe.isnull().values.any()

spam_test_dataframe.head()


# analyze the data 

spam_test_dataframe.describe()# this describe how our dataset looks like

spam_test_dataframe.groupby('label').describe() #this describe our lablel column

# Let us add another column called length of feature which will have how much does the message length is.
spam_test_dataframe['length'] = spam_test_dataframe['feature'].apply(len)
spam_test_dataframe.head()


# Data Visualization
spam_test_dataframe['length'].plot(bins=100, kind='hist')

spam_test_dataframe.length.describe()

# So the message with longest length is of 910 characters
spam_test_dataframe[spam_test_dataframe['length'] == 910]['feature'].iloc[0]

spam_test_dataframe.hist(column = 'length', by ='label', bins = 50 , figsize = (12,4))

#Machine Learning Step
import string
from nltk.corpus import stopwords

# text pre-processing
spam_test_dataframe['feature'] = spam_test_dataframe['feature'].str.replace('[^\w\s]','')
spam_test_dataframe['feature'] = spam_test_dataframe['feature'].apply(lambda x: " ".join(x.lower() for x in x.split()))
stop = stopwords.words('english')
spam_test_dataframe['feature'] = spam_test_dataframe['feature'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Check to make sure its working
spam_test_dataframe['feature'].head()


#Lets do something even better beside above techniques, lets shorten the terms to their stem form.

from nltk.stem import PorterStemmer
st = PorterStemmer()

spam_test_dataframe['feature'] = spam_test_dataframe['feature'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(spam_test_dataframe['feature'].values.astype('U'), spam_test_dataframe['label'], test_size=0.2, random_state=1)

print(len(X_train), len(X_test), len(y_train) + len(y_test))


# Naive Bayes
# Pipelining 
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB()),])
text_clf = text_clf.fit(X_train, y_train)
# using GridSearch CV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
gs_clf.best_score_
gs_clf.best_params_
predicted_nb = gs_clf.predict(X_test)
print(predicted_nb)


# Decision Tree
# Decisiton Tree Pipelining 
dt = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-dt', DecisionTreeClassifier(criterion = "gini", splitter="best",
                                                           max_depth=20, random_state = 42)),])
_ = dt.fit(X_train, y_train)

predicted_dt = dt.predict(X_test) 
print(predicted_dt)


#Random Forest
# Pipelining 
rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-rf', RandomForestClassifier(n_estimators = 100, max_depth=5, random_state = 42)),])
_ = rf.fit(X_train, y_train)

predicted_rf = rf.predict(X_test) 
print(predicted_rf)   


#Support Vector Machine
# using SVM
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)),])
_ = text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_test)
print(predicted_svm)


#Classification Report
from sklearn.metrics import classification_report
target_names = ['Features', 'Labels']

print(classification_report(y_test, predicted_nb, target_names=target_names))


print(classification_report(y_test, predicted_dt, target_names=target_names))


print(classification_report(y_test, predicted_rf, target_names=target_names))


print(classification_report(y_test, predicted_svm, target_names=target_names))


# Accuracy Score
precision_nb = accuracy_score(y_test, predicted_nb)
print("Naive Bayes Accuracy Score: ", precision_nb)

precision_dt = accuracy_score(y_test, predicted_dt)
print("Decision Tree Accuracy Score: ", precision_dt)


precision_rf = accuracy_score(y_test, predicted_rf)
print("Random Forest Accuracy Score: ", precision_dt)


precision_svm = accuracy_score(y_test, predicted_svm)
print("Support Vector Machine Accuracy Score: ", precision_dt)

highest = max(precision_nb, precision_dt, precision_rf, precision_svm)
print("the the highest accuracy is: ", highest)

