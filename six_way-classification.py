#!/usr/bin/env python
# coding: utf-8

# ##### List of Modules

# In[1]:


## pandas used to transform given tsv to csv file
import pandas as pd

## CountVectorizer tokenizes the collection of text documents and build a vocabulary of known words it returns ints
## TfidVectorizer is same as CountVectorizer but it returns float  values. In the below I'd compare the both values it false
## because float and int are not equal
##
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

## from sklearn.linear_model. I'd imported "PassiveAggressiveClassifier, SGDClassifier" for classification
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier

## from sklearn.svm. I'd imported "LinearSVC" for classication
from sklearn.svm import LinearSVC

## from sklearn.naive_bayes. I'd imported "MultinomialNB" is another kind of classifier most of the Data Scienctist use this
## one for better accuracy
from sklearn.naive_bayes import MultinomialNB

## metrics for predictions
from sklearn import metrics

## pyplot for visualizing graphs
import matplotlib.pyplot as plt

## shuffle for shuffling the data randomly
from sklearn.utils import shuffle

## classification_report which gives f1-score, precision, recall, support
from sklearn.metrics import classification_report

## seaborn is used for Data Visualizarion library
## I made confusion matrix for every classifier. So, for visualizing we need seaborn
import seaborn as sns


# Given data is .tsv extension. I had converting it into .csv file for training and testing using pandas



train2_tsv = 'train2.tsv'
train2_csv = pd.read_table(train2_tsv,sep = '\t')
train2_csv.to_csv('train2.csv',index = True)


# Let's see some train data



train = pd.read_csv('train2.csv')
train.head()


# Now data has been converted csv file. But it doesn't have column name. So, I had grab the what I need for training and then change the data into DataFrame using pandas



train_dict = {'ID':train['0'],'train_statement':train['Says the Annies List political group supports third-trimester abortions on demand.'],
          'Justification':train["That's a premise that he fails to back up. Annie's List makes no bones about being comfortable with candidates who oppose further restrictions on late-term abortions. Then again, this year its backing two House candidates who voted for more limits."],
          'train_label':train['false']}
train_df = pd.DataFrame(train_dict)


# Let's see the data now



train_df.head()


# Similarly as train data I had converted .tsv file extension into csv file



test2_tsv = 'test2.tsv'
test2_csv = pd.read_table(test2_tsv,sep = '\t')
test2_csv.to_csv('test2.csv',index = False)




test = pd.read_csv('test2.csv')
test.head()



test_dict = {'ID':test['0'],'test_statement':test['Building a wall on the U.S.-Mexico border will take literally years.'],
          'Justification':test['Meantime, engineering experts agree the wall would most likely take years to complete. Keep in mind, too, it took more than six years to build roughly 700 miles of fence and barriers along the roughly 2,000-mile U. S. -Mexico border.'],
          'test_label':test['true']}
test_df = pd.DataFrame(test_dict)





test_df.head()




## CountVectorizer tokenizes the collection of text documents and build a vocabulary of known words it returns ints
count_vectorizer = CountVectorizer(stop_words = 'english')

## function of fit_transform is fit and transform the function for feature extraction
count_train = count_vectorizer.fit_transform(train_df['train_statement'])

## transforms documents to document-type matrix
count_test = count_vectorizer.transform(test_df['test_statement'])



## TfidVectorizer is same as CountVectorizer but it returns float  values. In the below I'd compare the both values it false
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

## function of fit_transform is fit and transform the function for feature extraction
tfidf_train = tfidf_vectorizer.fit_transform(train_df['train_statement'])

## transforms documents to document-type matrix
tfidf_test = tfidf_vectorizer.transform(test_df['test_statement'])


# Let's see the feature names names and their matrix values



print(tfidf_vectorizer.get_feature_names()[:10])
print(tfidf_train.A[:5])


# Below code let you know that CountVectorizer rerurns integer and TfidfVectorizer returns the float values values for the same dataset. Finally I'd compare the two values.



count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
print(count_df.head())
print(tfidf_df.head())
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)
print(count_df.equals(tfidf_df))


# ## Multinomial Naive Bayes Classifier



## Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

## Fit the classifier to the training data
nb_classifier.fit(count_train, train_df['train_label'])

## Create the predicted tags: pred
mnb_pred = nb_classifier.predict(count_test)

# Create the predicted tags: pred
mnb_score = metrics.accuracy_score(test_df['test_label'], mnb_pred)

# Calculate the confusion matrix: mnb_cm
mnb_cm = metrics.confusion_matrix(test_df['test_label'], mnb_pred, labels=['true', 'false'])
print('Confusion Matrix --- Multinomial Naive Bayes')
print(mnb_cm)
print("Multinomial Naive Bayes classifier accuracy:   %0.3f" % mnb_score)


# Below code shows the confusion matrix in a graphical form for Multinomial Naive Bayes classifier and classification reports



mnb_cm = metrics.confusion_matrix(test_df['test_label'],mnb_pred)
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(mnb_cm, annot=True, linewidth=.5, square = True, cmap = 'Blues_r',fmt='f');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix', size = 15);

report = classification_report(test_df['test_label'],mnb_pred)
print(report)


# ## Passive Aggressive Classifier



## Instantiating a Passive Aggressive Classifier : pa_tfidf_clf
pa_tfidf_clf = PassiveAggressiveClassifier()

## Fit the classifier to the training data
pa_tfidf_clf.fit(count_train, train_df['train_label'])

## Create the predicted tags: pac_pred
pac_pred = pa_tfidf_clf.predict(count_test)
## Calculate the accuracy score: pac_score
pac_score = metrics.accuracy_score(test_df['test_label'], pac_pred)

## Calculate the confusion matrix: pac_cm
pac_cm = metrics.confusion_matrix(test_df['test_label'], pac_pred, labels=['true', 'false'])
print('Confusion Matrix --- PassiveAggressiveClassifier')
print(pac_cm)
print("Passive Aggressive Classifier accuracy:   %0.3f" % pac_score)


# Below code shows the confusion matrix in a graphical form for PassiveAggressiveClassifier and classification reports



pac_cm = metrics.confusion_matrix(test_df['test_label'],pac_pred)
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(pac_cm, annot=True, linewidth=.5, square = True, cmap = 'Blues_r',fmt='f');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix', size = 15);

report = classification_report(test_df['test_label'],pac_pred)
print(report)


# ## Support Vector Classifier



## Instantiate a Support Vector classifier: svc_tfidf_clf
svc_tfidf_clf = LinearSVC()

## Fit the classifier to the training data
svc_tfidf_clf.fit(count_train, train_df['train_label'])

## Create the predicted tags: svc_pred
svc_pred = svc_tfidf_clf.predict(count_test)

## Calculate the accuracy score: svc_score
svc_score = metrics.accuracy_score(test_df['test_label'], svc_pred)

## Calculate the confusion matrix: cm
svc_cm = metrics.confusion_matrix(test_df['test_label'], svc_pred, labels=['true', 'false'])
print('Confusion Matrix --- LinearSVC')
print(svc_cm)
print("Support Vector classifier accuracy:   %0.3f" % svc_score)


# Below code shows the confusion matrix in a graphical form for SVCClassifier and classification reports



svc_cm = metrics.confusion_matrix(test_df['test_label'],svc_pred)
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(svc_cm, annot=True, linewidth=.5, square = True, cmap = 'Blues_r',fmt='f');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix', size = 15);

report = classification_report(test_df['test_label'],svc_pred)
print(report)


# ## Stochastic Gradient Descent Classifier



## Instantiate a Stochastic Gradient Descent: sgd_tfidf_clf
sgd_tfidf_clf = SGDClassifier()

## Fit the classifier to the training data
sgd_tfidf_clf.fit(count_train, train_df['train_label'])

## Create the predicted tags: sgd_pred
sgd_pred = sgd_tfidf_clf.predict(count_test)

## Calculate the accuracy score: score
sgd_score = metrics.accuracy_score(test_df['test_label'], sgd_pred)

## Calculate the confusion matrix: cm
sgd_cm = metrics.confusion_matrix(test_df['test_label'], sgd_pred, labels=['true', 'false'])
print('Confusion Matrix --- SGD Classifier')
print(sgd_cm)

print("Stochastic Gradient Descent Classifier accuracy:   %0.3f" % sgd_score)


# Below code shows the confusion matrix in a graphical form for SGDClassifier and classification reports
#



sgd_cm = metrics.confusion_matrix(test_df['test_label'],sgd_pred)
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(sgd_cm, annot=True, linewidth=.5, square = True, cmap = 'Blues_r',fmt='f');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix', size = 15);

report = classification_report(test_df['test_label'],sgd_pred)
print(report)
