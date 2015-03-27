from sklearn import tree
import numpy as np


target=np.array([1,2,3,0,1,1,3,1])
input=np.array([[1,2,3,4],[2,3,4,4],[1,1,3,2],[1,3,1,4],[1,2,3,4],[2,3,4,4],[1,1,3,2],[1,3,1,4]])

#input=input.T
print input

Tree=tree.DecisionTreeClassifier()#default setting
fitTree=Tree.fit(input,target)
print fitTree.predict([1.,2.,1.,4.])
#print the corresponding probabilities
print fitTree.predict_proba([1.,2.,1.,4.])#prediction with prob


from sklearn.externals.six import StringIO
import pydot
dot_data=StringIO()
