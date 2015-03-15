import json
import time
import re
import pickle
from stemming.porter2 import stem

def read_yelp(idxer):
    fp = open("../data/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json")
    line = fp.readline()
    i = 0
    while line and i < 1000:
        line = fp.readline()
        #try:
        review = json.loads(line)
        idxer.addDoc(review['text'])
        #except:
        #    print "parse", i, "failed"
        i += 1
        if i % 1000 == 0:
            print i

class Idxer:
    def __init__(self):
        self.docId = 0
        self.rgx = re.compile("[\.\?\n\!]")
        self.invertedIdx = {}
        self.docLs = []
    def addDoc(self, doc):
        ri = self.docId
        self.docId += 1
        sentenceLs = self.rgx.split(doc)
        self.docLs.append([])
        for si, sentence in enumerate(sentenceLs):
            wordLs = sentence.strip().lower().split()
            for word in wordLs:
                word = stem(word)
                if word not in self.invertedIdx:
                    self.invertedIdx[word] = []
                # the si-th sentence in ri-th review is idexed by ri*1000+si
                self.invertedIdx[word].append(ri*1000 + si) 
            self.docLs[ri].append(wordLs)




if __name__ == "__main__":
    start_time = time.time()
    idxer = Idxer()
    read_yelp(idxer)
    #print idxer.invertedIdx
    while True:
        word = raw_input('Search?')
        word = stem(word.strip().lower())
        print word
        if word in idxer.invertedIdx:
            docLs = idxer.invertedIdx[word]
            for doc in docLs:
                ri = doc/1000
                si = doc%1000
                print " ".join(idxer.docLs[ri][si])


