#importing SVM from sklearn package
from sklearn import svm
#import accuracy_score from sklearn.metrices to find accuracy
from sklearn.metrics import accuracy_score
#giving data set for train our SVM
X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
#intilizing our svm using clf(classfier) variable
clf=svm.SVC()
#train our svm using fit function
clf=clf.fit(X_train,Y_train)
#providing unlabled data point

X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
#providing correct prediction for the above unlabled datapoint
Y_test=['male','male','male','female','female','female','male','male']
#storing the result of prediction into a variable
Y_prediction=clf.predict(X_test)
#printing the prediction and accuracy of our SVM
print("prediction of SVM:",Y_prediction)
print("accuracy of SVM:",accuracy_score(Y_test,Y_prediction))