#import tree from sklearn
from sklearn import tree
#import accuracy_score from sklearn.metrices
from sklearn.metrics import  accuracy_score
#input training data sets
X_train = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y_train = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
#intilizing dicision tree through clf(classifier) variable which store our decision tree
clf=tree.DecisionTreeClassifier()
#train our decision tree using fit() function
clf=clf.fit(X_train,Y_train)
#provide unlabled datapoint
X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
#provide accurate result of unlabled datapoint so that we can check accuracy in last
Y_test=['male','male','male','female','female','female','male','male']
#store result of prediction in a variable
Y_prediction=clf.predict(X_test)
#print prediction resultand accuracy of our decision tree
print("Prediction of Decision tree:",Y_prediction)
print("accuracy of Decision tree:",accuracy_score(Y_test,Y_prediction))