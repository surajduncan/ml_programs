# Write a program to implement k-Nearest Neighbor algorithm to classify the iris data set.
# Print both correct and wrong predictions. 
# Java/Python ML library classes can be used for this problem.






from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris=datasets.load_iris()
print("iris data set is loaded-----------------------")
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.1)
print("dataset is spilt into training and testing:")
print("size of trsining data and its label:",x_train.shape,y_train.shape)
print("size of testing data and its label:",x_test.shape,y_test.shape)
for i in range(len(iris.target_names)):
    print("lable",i,"_",str(iris.target_names[i]))
classifier=KNeighborsClassifier(n_neighbors=1)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print("results of classfication using knnwith k=1")
for r in range(0,len(x_test)):
    print("sample:",str(x_test[r]),"actual-label:",str(y_test[r]),"predicted_label:",str(y_pred[r]))
print("classification accuracy:",classifier.score(x_test,y_test))

