# Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task. 
# Built-in Java classes/API can be used to write the program. 
# Calculate the accuracy, precision, and recall for your data set.










import pandas as pd
msg=pd.read_csv('naivebuiltin.csv',header=None,names=['message','label'],)
print('The dimensions of the dataset',msg.shape)
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
y=msg.labelnum


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=1)
print('dimensions of train and test sets')
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

#output of count vectoriser is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)

print('\nclassification results of testing samples are given below')
for doc,p in zip(xtest,predicted):
    pred='pos' if p==1 else 'neg'
    print('%s->%s' %(doc,pred))
#printing accuracy metrics
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Recall and Precison ')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))
