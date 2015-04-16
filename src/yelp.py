import json
import time
import re
import pickle
import numpy as np
from collections import Counter
from stemming.porter2 import stem
import random
import os.path
from sklearn import tree

def load_stop_word():
    stopWord = {u'': True}
    fp = open("./stopword")
    line = fp.readline()
    while line:
        line = stem(line.strip())
        stopWord[line] = True
        line = fp.readline()
    return stopWord

def yelp_load(hasVotes = True):
    start_time = time.time()
    fin = open("../data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json")
    line = fin.readline()
    reviewLs = []
    i = 0
    while line :
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
    print "yelp_load() spend", time.time() - start_time, "to load", len(reviewLs), "reviews"
    return reviewLs

def yelp_select(reviewLs, rate):
    numSel = int(len(reviewLs)*rate)
    random.shuffle(reviewLs)
    return reviewLs[:numSel]

def yelp_process(reviewLs, stopWordDic):
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

    print "yelp_process() spend", time.time() - start_time, "to process", len(reviewLs), "reviews"
    return data, dictionary

# sort the word according to frequency
def gen_top_word(num, dictionary):
    topWordDic = {}
    topWordLs = sorted(dictionary.items(), key = lambda item: item[1][0], reverse = True)
    topWordLs = topWordLs[:num]
    for word, freq in topWordLs:
        topWordDic[word] = freq
    return topWordDic

# use transfer data's word count dic to feature vector by topWordDic
def dic_to_vec(topWordDic, data):
    start_time = time.time()
    # dataMatrix[i][-1] stores the star rates
    # dataMatrix[i][:-1] stores the feature vector
    dataMatrix = np.zeros((len(data), len(topWordDic) + 1))
    for idx, item in enumerate(data):
        star = item[0]
        wordCnt = item[1]
        dataMatrix[idx][-1] = star
        for widx, word in enumerate(topWordDic.keys()):
            if word in wordCnt:
                dataMatrix[idx][widx] = wordCnt[word]
    print "dic_to_vec() spend", time.time() - start_time, "to process", len(data), "data"
    return dataMatrix
        

# grouping data according to star rates
def yelp_group_data(dataMatrix):
    start_time = time.time()
    groupLs = [[],[],[],[],[]]
    for item in dataMatrix:
        groupLs[int(item[-1])].append(item)
    print "yelp_group_data() spend", time.time() - start_time, "to process", len(dataMatrix), "data"
    return groupLs

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


#separate trainingg/validation/testing data
def yelp_devide_data(groupLs, rateLs):
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
    print "yelp_devide_data() spend", time.time() - start_time
    return np.array(trainLs), np.array(validLs), np.array(testLs)


if __name__ == "__main__":

    dataFile = "../data/dataMatrix"
    if os.path.isfile(dataFile) is False:
    #if True:
        reviewLs = yelp_load()
        reviewLs = yelp_select(reviewLs, 0.04)
        #print len(reviewLs)
        
        stopWordDic = load_stop_word()
        #print stopWordDic
        
        data, dictionary = yelp_process(reviewLs, stopWordDic)
        #print data
        #print dictionary

        topWordDic = gen_top_word(1000, dictionary)
        #print topWordDic

        dataMatrix = dic_to_vec(topWordDic, data)
        #print dataMatrix
        pickle.dump( dataMatrix, open( dataFile, "wb" ) )
    else:
        dataMatrix = pickle.load(open(dataFile, 'rb'))

    for data in dataMatrix:
        if data[-1] > 2: # seperate the data by yelp_star_rate (2 + 1) -- in yelp_process() we minus the star_rate by 1 for convenience
            data[-1] = 1
        else:
            data[-1] = 0
    
    groupLs = yelp_group_data(dataMatrix)
    #for group in groupLs:
    #    print group

    trainLs, validLs, testLs = yelp_devide_data(groupLs, [.85, .0, .15])
    #print trainLs

    start_time = time.time()
    fitTree = treeCf(trainLs[:,:-1], trainLs[:,-1]);

    predClassTest=fitTree.predict(testLs[:,:-1])# predicted class
    predClassTrain=fitTree.predict(trainLs[:,:-1])# predicted class

    print time.time() - start_time

    print "Feature Dimension:", len(testLs[0])-1
    print "Train Num:", len(trainLs)
    print "Test Num:", len(testLs)
    
    cnt = 0
    goodCnt = 0
    for pc, c in zip(predClassTest, testLs[:,-1]):
        if pc == c:
            cnt += 1
        if c == 1:
            goodCnt += 1
    print "Test Accuracy:", cnt * 1.0/ len(predClassTest)
    print "Test goodCnt rate:", goodCnt * 1.0 / len(predClassTest)

    cnt = 0
    goodCnt = 0
    for pc, c in zip(predClassTrain, trainLs[:,-1]):
        if pc == c:
            cnt += 1
        if c == 1:
            goodCnt += 1
    print "Train Accuracy:", cnt * 1.0/ len(predClassTrain)
    print "Train goodCnt rate:", goodCnt * 1.0 / len(predClassTrain)








