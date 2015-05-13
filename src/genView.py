import json

def loadRec(fileName):
    fp = open(fileName)
    objLs = []

    line = fp.readline()
    i = 0
    while line:
        objLs.append(json.loads(line))
        #print objLs[i]
        line = fp.readline()
        i+=1
    return objLs

def sortRec(objLs):
    #objLs = sorted(objLs, key = lambda obj: obj["MLType"]) 
    priorLs = ["trainNum", "MLType"]
    objLs = sorted(objLs, key = lambda obj:obj["MLType"])
    objLs = sorted(objLs, key = lambda obj:obj["trainNum"])
    return objLs


def genHtml(objLs, htmlFile):
    table = ["tfidf", "stopWord", "trainNum", "wordDim", "MLType", "testAccuracy", "trainAccuracy", "trainTime", "predTrainTime"]
    fp = open(htmlFile, 'w')
    fp.write('<table style="width:100%>"\n')
    fp.write("<tr>\n")
    for item in table:
        fp.write("<td>")
        fp.write(item)
        fp.write("</td>")
    fp.write("</tr>\n")
    for obj in objLs:
        fp.write("<tr>\n")
        for item in table:
            fp.write("<td>")
            if(item in obj):
                if type(obj[item]) is float:
                    obj[item] = round(obj[item], 4)
                fp.write(str(obj[item]))
            else:
                fp.write("###")
            fp.write("</td>")
        fp.write("</tr>\n")

    fp.write("<table>\n")



if __name__ == "__main__":
    objLs = loadRec("../data/rec.txt")
    objLs = sortRec(objLs)
    genHtml(objLs, "../data/rec.html")


