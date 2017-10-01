#importing neighbor from sklearn
from sklearn import neighbors
#importing accuracy_score from metrices which used to find accuracy of our algo
from sklearn.metrics import accuracy_score
#providing training datasets to our algo
X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
#intilizing the kNeighborclassifier from neighbor
clf=neighbors.KNeighborsClassifier()
#training our algo using fit() function
clf=clf.fit(X_train,Y_train)
#providing unlabled datapoint

X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
#providing correct prediction of unlabled datapoint so that we can check accuracy of our algo in last
Y_test=['male','male','male','female','female','female','male','male']
#storing result of prediction into a variable
Y_prediction=clf.predict(X_test)
#printing prediction result and accuracy of our algo
print("prediction by kneighbor:",Y_prediction)
print("accuracy of Kneighbors:",accuracy_score(Y_test,Y_prediction))