import json
import time
import re
import pickle
import numpy as np
from collections import Counter
from stemming.porter2 import stem
import random
import math
import os.path
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
from copy import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

class Yelp:
    def __init__(self, rec):
        self.rec = rec

    def load_stop_word(self):
        stopWord = {u'': True}
        fp = open("./stopword")
        line = fp.readline()
        while line:
            line = stem(line.strip())
            stopWord[line] = True
            line = fp.readline()
        return stopWord
    
    def load(self, hasVotes = True):
        start_time = time.time()
        fin = open("../data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json")
        line = fin.readline()
        reviewLs = []
        i = 0
        while line:
            try:
                review = json.loads(line)
                if hasVotes :
                    votes = review["votes"]
                    if votes["funny"] + votes["useful"] + votes["cool"] > 1:
                        reviewLs.append(review)
                else:
                    reviewLs.append(review)
            except:
                print "parse", i, "failed"
    
            line = fin.readline()
            i += 1
            if i % 10000 == 0:
                print i
        fin.close()
        print "load() spend", time.time() - start_time, "to load", len(reviewLs), "reviews"
        self.rec["load_time"] = time.time() - start_time
        self.rec["hasVotes"] = hasVotes
        return reviewLs
    
    def select(self, reviewLs, rate):
        numSel = int(len(reviewLs)*rate)
        random.shuffle(reviewLs)
        return reviewLs[:numSel]
    
    def process(self, reviewLs, stopWordDic):
        start_time = time.time()
        rgx = re.compile("[\`\~\!\@\#\$\%\^\&\*\(\)\-\_\=\+\;\:\"\,\<\.\>\/\?\ \n]")
        data = [] # [(star_rate_1, word_counter_1), (star_rate_2, word_counter_2) ...]
        dictionary = {} # global dictionary {"word_1": (corpus_freq1, doc_freq1)...}
        for review in reviewLs:
            #lower case, tokenize
            wordLs  = rgx.split(review["text"].lower())
    
            #stemming
            wordLs = [stem(word) for word in wordLs]
    
            #generate word counter (a dictionary)
            wordCnt = Counter(wordLs)
    
            #remove stop word and generate global dictionary
            for word, freq in wordCnt.items():
                if word in stopWordDic:
                    del wordCnt[word]
                else:
                    if word in dictionary:
                        dictionary[word][0] += freq # count the total appearing time in whole review
                        dictionary[word][1] += 1 #count the document number that the word appears in
                    else:
                        dictionary[word] = [freq,1]

            data.append((review["stars"]-1, wordCnt)) # minus the star_rate by 1 for convenience

        # transfer cnt to idf
        docNum = len(reviewLs)
        for word, stat in dictionary.items():
            cnt = stat[1]
            idf = math.log(docNum*1.0/(cnt+1))
            stat[1] = idf
    
    
        self.rec["process_time"] = time.time() - start_time
        print "process() spend", time.time() - start_time, "to process", len(reviewLs), "reviews"
        return data, dictionary
    
    # sort the word according to frequency
    def gen_top_word(self, num, dictionary):
        topWordDic = {}
        topWordLs = sorted(dictionary.items(), key = lambda item: item[1][0], reverse = True)
        topWordLs = topWordLs[:num]
        widx = 0
        for word, freq in topWordLs:
            topWordDic[word] = (widx, freq)
            widx += 1
        return topWordDic
    
    # use transfer data's word count dic to feature vector by topWordDic
    def dic_to_vec(self, topWordDic, data):
        start_time = time.time()
        # dataMatrix[i][-1] stores the star rates
        # dataMatrix[i][:-1] stores the feature vector
        dataMatrix = np.zeros((len(data), len(topWordDic) + 1))
        for idx, item in enumerate(data):
            star = item[0]
            wordCnt = item[1]
            dataMatrix[idx][-1] = star
            for word, freq in wordCnt.items():
                if word in topWordDic:
                    widx = topWordDic[word][0]
                    dataMatrix[idx][widx] = wordCnt[word]
        self.rec["dic_to_vec_time"] = time.time() - start_time
        print "dic_to_vec() spend", time.time() - start_time, "to process", len(data), "data"
        return dataMatrix

    def dic_to_tfidf_vec(self, topWordDic, dictionary, data):
        start_time = time.time()
        # dataMatrix[i][-1] stores the star rates
        # dataMatrix[i][:-1] stores the feature vector
        dataMatrix = np.zeros((len(data), len(topWordDic) + 1))
        for idx, item in enumerate(data):
            star = item[0]
            wordCnt = item[1]
            dataMatrix[idx][-1] = star

            docWordNum = 0
            for word, freq in wordCnt.items():
                if word in topWordDic:
                    docWordNum += wordCnt[word]

            for word, freq in wordCnt.items():
                if word in topWordDic:
                    widx = topWordDic[word][0]
                    tf = wordCnt[word]*1.0/docWordNum
                    idf = dictionary[word][1]
                    dataMatrix[idx][widx] = tf*idf
        self.rec["tfidf"] = True
        self.rec["dic_to_tfidf_vec_time"] = time.time() - start_time
        print "dic_to_tfidf_vec() spend", time.time() - start_time, "to process", len(data), "data"
        return dataMatrix
            
    
    # grouping data according to star rates
    def group_data(self, dataMatrix):
        start_time = time.time()
        groupLs = [[],[],[],[],[]]
        for item in dataMatrix:
            groupLs[int(item[-1])].append(item)
        self.rec["group_data_time"] = time.time() - start_time
        print "group_data() spend", time.time() - start_time, "to process", len(dataMatrix), "data"
        return groupLs

    #separate training/validation/testing data
    def devide_data(self, groupLs, rateLs):
        start_time = time.time()
        trainLs = []
        validLs = []
        testLs = []
    
    
        for group in groupLs:
            num = len(group)
    
            trainNum = int(rateLs[0] * num)
            validNum = int(rateLs[1] * num)
            testNum  = num - trainNum - validNum

            trainLs += group[:trainNum]
            validLs += group[trainNum: trainNum + validNum]
            testLs += group[trainNum + validNum:]
        print "devide_data() spend", time.time() - start_time

        self.rec["trainNum"] = len(trainLs)
        self.rec["validNum"] = len(validLs)
        self.rec["testNum"] = len(testLs)
    
        self.rec["devide_data_time"] = time.time() - start_time
        return np.array(trainLs), np.array(validLs), np.array(testLs)

    def pipeline(self, tfidf = False, stopWord = True):
        if tfidf:
            if stopWord is False:
                dataFile = "../data/dataMatrix_tfidf_NoStopWord"
                recFile = "../data/rec_tfidf_NoStopWord"
            else:
                dataFile = "../data/dataMatrix_tfidf"
                recFile = "../data/rec_tfidf"
        else:
            dataFile = "../data/dataMatrix"
            recFile = "../data/rec"
        if os.path.isfile(dataFile) is False:
        #if True:
            reviewLs = self.load()
            reviewLs = self.select(reviewLs, self.rec["selectRate"])
            #print len(reviewLs)
            
            if stopWord:
                stopWordDic = self.load_stop_word()
                #print stopWordDic
            else:
                self.rec["stopWord"] = False
                stopWordDic = {u'': True}
            
            data, dictionary = self.process(reviewLs, stopWordDic)
            #print data
            #print dictionary

            topWordDic = self.gen_top_word(self.rec['wordDim'], dictionary)
            #print topWordDic

            if tfidf:
                dataMatrix = self.dic_to_tfidf_vec(topWordDic, dictionary, data)
                #print dataMatrix
            else:
                dataMatrix = self.dic_to_vec(topWordDic, data)
                #print dataMatrix

            pickle.dump( dataMatrix, open( dataFile, "wb" ) )
            pickle.dump( self.rec, open( recFile, "wb" ) )
        else:
            dataMatrix = pickle.load(open(dataFile, 'rb'))
            self.rec = pickle.load(open(recFile, 'rb'))

        for data in dataMatrix:
            if data[-1] > 2: 
                # seperate the data by yelp_star_rate (2 + 1) -- in process() we minus the star_rate by 1 for convenience
                data[-1] = 1
            else:
                data[-1] = 0
        
        groupLs = self.group_data(dataMatrix)
        #for group in groupLs:
        #    print group

        trainLs, validLs, testLs = self.devide_data(groupLs, self.rec["dataDistr"])

        return trainLs, validLs, testLs


def treeCf(Xtrain, Ytrain):
    """fit tree with trainning set
    """
    Tree=DecisionTreeClassifier()#default setting
    fitTree=Tree.fit(Xtrain,Ytrain)
    return fitTree

def treePredict(Xtrain,Ytrain, Xtest):
    """predict class with testing data"""
    fitTree=treeCf(Xtrain,Ytrain)
    classProb=fitTree.predict_proba(Xtest) #class with prob
    predClass=fitTree.predict(Xtest)# predicted class
    return predClass, classProb

def getAccuracy(realClass, predClass):
    cnt = 0
    goodCnt = 0
    for pc, c in zip(predClass, realClass):
        if pc == c:
            cnt += 1
        if c == 1:
            goodCnt += 1
    accuracy = cnt * 1.0/ len(predClass)
    classRate = goodCnt * 1.0 / len(predClass)
    return accuracy, classRate

def getError(realClass, predClass):
    accuracy, classRate = getAccuracy(realClass, predClass)
    return 1 - accuracy, classRate



def testTreeDepth(MLType, trainLs, testLs, rec):
    """generate the tree depth that minimize the test error"""
    trainErrorList=[]
    testErrorList=[]
    depthlist=[]
    copyRec = copy(rec)
    for depth in range(1,80,10):
        rec = copy(copyRec)
        starttime=time.time()
        model = ""
        if MLType == "decision tree":
            rec["MLType"] = "decision tree"
            rec["max_depth"] = depth
            Tree=DecisionTreeClassifier(max_depth=depth)
            model = Tree.fit(trainLs[:,:-1],trainLs[:,-1])
        else: #random forest
            rec["MLType"] = "random forest"
            rec["max_depth"] = depth
            n_estimators = 100
            rec["n_estimators"] = n_estimators
            forest = RandomForestClassifier(n_estimators = n_estimators, max_depth = depth)
            model = forest.fit(trainLs[:,0:-1],trainLs[:,-1])

        trainAccuracy, testAccuracy = modelTest(model, trainLs, testLs, rec)
        errorTrain = 1-trainAccuracy
        errorTest  = 1-testAccuracy

        depthlist.append(depth)
        trainErrorList.append(errorTrain)
        testErrorList.append(errorTest)

        print 'depth:' ,depth
        print 'Train error rate:', errorTrain
        print 'Test error rate:', errorTest
        print 'Run time:', time.time()-starttime

    plt.plot(depthlist,trainErrorList,depthlist,testErrorList)
    plt.legend(['train error','test error'])
    plt.xlabel('Depths')
    plt.show()

def modelTest(model, trainLs, testLs, rec):
    #predict train
    start_time = time.time()
    predClassTrain=model.predict(trainLs[:,:-1])# predicted class
    rec["predTrainTime"] = time.time() - start_time

    #train accuracy
    trainAccuracy, trainClassRate = getAccuracy(trainLs[:,-1], predClassTrain)
    rec["trainAccuracy"]  = trainAccuracy
    rec["trainClassRate"] = trainClassRate

    #predict test
    start_time = time.time()
    predClassTest=model.predict(testLs[:,:-1])# predicted class
    rec["predTestTime"] = time.time() - start_time

    #test accuracy
    testAccuracy, testClassRate = getAccuracy(testLs[:,-1], predClassTest)
    rec["testAccuracy"]  = testAccuracy
    rec["testClassRate"] = testClassRate

    print "trainClassRate:", rec["trainClassRate"]
    print "trainAccuracy:", rec["trainAccuracy"]
    print "testClassRate:", rec["testClassRate"]
    print "testAccuracy:", rec["testAccuracy"]

    fp = open("../data/rec.txt", 'a')
    fp.write(json.dumps(rec))
    fp.write("\n")

    return trainAccuracy, testAccuracy

def testDecisionTree(trainLs, testLs, rec):
    copyRec = copy(rec)
    rec["MLType"] = "decision tree"
    start_time = time.time()
    fitTree = treeCf(trainLs[:,:-1], trainLs[:,-1]);
    rec["trainTime"] = time.time() - start_time

    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]
    
    modelTest(fitTree, trainLs, testLs, rec)



def testRandomForest(trainLs, testLs, rec):
    copyRec = copy(rec)
    rec["MLType"] = "random forest"
    n_estimators = 100
    rec["n_estimators"] = n_estimators

    start_time = time.time()
    forest = RandomForestClassifier(n_estimators = n_estimators)
    forest = forest.fit(trainLs[:,0:-1],trainLs[:,-1])
    rec["trainTime"] = time.time() - start_time

    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]

    modelTest(forest, trainLs, testLs, rec)

def testLogisticRegression(trainLs, testLs, rec):
    copyRec = copy(rec)
    rec["MLType"] = "logistic regression"
    rec["L2"] = 1.0

    start_time = time.time()
    logreg = LogisticRegression(C=rec["L2"])
    model = logreg.fit(trainLs[:,0:-1], trainLs[:,-1])
    rec["trainTime"] = time.time() - start_time

    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]

    modelTest(model , trainLs, testLs, rec)

def testSVM(trainLs, testLs, rec):
    copyRec = copy(rec)
    rec["MLType"] = "svm"

    start_time = time.time()
    clf = svm.SVC()
    model = clf.fit(trainLs[:,0:-1], trainLs[:,-1])
    rec["trainTime"] = time.time() - start_time

    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]

    modelTest(model , trainLs, testLs, rec)

def testKNN(trainLs, testLs, rec):
    copyRec = copy(rec)
    rec["MLType"] = "knn"

    start_time = time.time()
    neigh = KNeighborsClassifier(n_neighbors=5)
    model = neigh.fit(trainLs[:,0:-1], trainLs[:,-1])
    rec["trainTime"] = time.time() - start_time

    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]

    modelTest(model , trainLs, testLs, rec)

def testLinearRegression(trainLs, testLs, rec):
    copyRec = copy(rec)
    rec["MLType"] = "linear regression"

    start_time = time.time()
    lg = LinearRegression()
    model = lg.fit(trainLs[:,0:-1], trainLs[:,-1])
    rec["trainTime"] = time.time() - start_time

    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]

    modelTest(model , trainLs, testLs, rec)

def testAdaBoost(trainLs, testLs, rec):
    copyRec = copy(rec)
    rec["MLType"] = "ada boost"
    n_estimators = 100
    rec["n_estimators"] = n_estimators
    rec["algorithm"] = "SAMME.R"
    rec["max_depth"] = 10

    start_time = time.time()
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = rec["max_depth"]),
                             algorithm=rec["algorithm"],
                             n_estimators=n_estimators)
    model = bdt.fit(trainLs[:,0:-1], trainLs[:,-1])
    rec["trainTime"] = time.time() - start_time

    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]

    modelTest(model , trainLs, testLs, rec)



def Svm(trainLs,testLs,rec):
    copyRec=copy(rec)
    rec["MLType"] = "Support Vector Machine"
    rec['Lambda/C']=5
    
    start_time = time.time()
    clf = svm.SVC(kernel='rbf',C=5)#Lambda 5 is the best based on OptLambda_svm
    svm_fit=clf.fit(trainLs[:,0:-1],trainLs[:,-1])
    rec["trainTime"] = time.time() - start_time
    
    print "MLType", rec["MLType"]
    print "trainTime", rec["trainTime"]
    
    modelTest(svm_fit,trainLs,testLs,rec)

def TestSvm_Lambda(trainLs, testLs, rec):
    """search for penalty parameter for svm that minimize the test error"""
    trainErrorList=[]
    testErrorList=[]
    LambdaList=[]
    copyRec = copy(rec)
    #for i in range(-3,3):#search in big scale 
        #Lambda = 10**i
    for i in range(1,12,2):
    #for i in [4.2,4.5,4.8,5.1,5.4,5.7]:#then narrow down: but dosesn't make a different 
        Lambda=i
        rec = copy(copyRec)
        starttime=time.time()
        model = ""

        rec["MLType"] = "svm"
        rec["penalty:Lambda"] = Lambda
        clf = svm.SVC(kernel='rbf',C=Lambda)
        model = clf.fit(trainLs[:,:-1],trainLs[:,-1])
        
        trainAccuracy,testAccuracy = modelTest(model,trainLs,testLs,rec)
        errorTrain = 1-trainAccuracy
        errorTest  = 1-testAccuracy

        LambdaList.append(math.log(Lambda,10))
        trainErrorList.append(errorTrain)
        testErrorList.append(errorTest)

        print'Lambda:', Lambda
        print 'Train error rate:', errorTrain
        print 'Test error rate:', errorTest
        print 'Run time:', time.time()-starttime
        print '\n'

    plt.plot(LambdaList,trainErrorList,LambdaList,testErrorList)
    plt.legend(['train error','test error'])
    plt.xlabel('logLambda')
    plt.show()



if False:
#if __name__ == "__main__":
    rec = {};
    rec['selectRate'] = .04
    rec['wordDim'] = 2000
    rec['dataDistr'] = [.85, .0, .15]
    yelp = Yelp(rec);

    trainLs, validLs, testLs = yelp.pipeline(tfidf = True, stopWord = False)
    rec = yelp.rec;

    #testDecisionTree(trainLs, testLs, rec)

    #testRandomForest(trainLs, testLs, rec)

    #testTreeDepth("random forest", trainLs, testLs, rec)  

    #testLogisticRegression(trainLs, testLs, rec)
    
    #TestSvm_Lambda(trainLs, testLs, rec)
    Svm(trainLs,testLs,rec)



