from sklearn import tree
import numpy as np


Ytrain=np.array([1,2,3,0,1,1,3,1])
Xtrain=np.array([[1,2,3,4],[2,3,4,4],[1,1,3,2],[1,3,1,4],[1,2,3,4],[2,3,4,4],[1,1,3,2],[1,3,1,4]])
Xtest=[1.,2.,1.,4.]

def treeCf(Xtrain, Ytrain):
    """fit tree with trainning set
    """
    Tree=tree.DecisionTreeClassifier()#default setting
    fitTree=Tree.fit(Xtrain,Ytrain)
    return fitTree

def treePredict(Xtrain,Ytrain, Xtest):
    """predict class with testing data"""
    fitTree=treeCf(Xtrain,Ytrain)
    classProb=fitTree.predict_proba(Xtest) #class with prob
    predClass=fitTree.predict(Xtest)# predicted class
    return predClass, classProb

predClass,classProb= treePredict(Xtrain,Ytrain,Xtest)
print predClass
print classProb 

## visualized the tree 
from sklearn.externals.six import StringIO
import pydot
dot_data=StringIO()
out=tree.export_graphviz(treeCf(Xtrain,Ytrain),out_file=dot_data)
treeGraph=pydot.graph_from_dot_data(dot_data.getvalue())
# writing to pdf file 
treeGraph.write_pdf("tree.pdf") 
