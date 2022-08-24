"""
Step 1:
    在篱笆型(Trellis)网络中，列出每个观测值所有可能的状态（也就是音对应的所有可能的字）
Step 2:
    统计每个节点的词频，创建一个netDict，元素为字典的列表，值为该字出现的频率
Step 3:
    计算拼音转换汉字的统计概率 
Step 4:
    每两层之间创建状态转移矩阵
Step 5:
    使用viterbi算法计算最优路径
Step 6:
    回溯，展示结果

问题：
    如果前一个层的词的词频在语料库中找不到怎么办  step4 getprobability
"""

import math

import numpy as np
import os
from unittest import result
import dill
from pyparsing import Word
from pypinyin import pinyin,Style

from torch import le


class Step1:
    def __init__(self,pronounces : list, lexiconFile : str):
        """
        Parameter(s):
            pronounce: list
                观测值序列，每个拼音为字符串，所有拼音组成一个列表
            lexiconFile: file
                拼音文件
        """
        self.net = [[] for _ in range(len(pronounces))]  # 状态的网络，每个列表代表一列
        self.main(pronounces,lexiconFile)

    def main(self,sentencePron: list, lexiconFile: str) -> None:
        """列出每个观测值所有可能的状态"""

        with open(lexiconFile, encoding='gbk') as F:
            for line in F:
                word,Wordpron = line.split('\t')
                lenWord = len(word)
                Wordpron = [x[:-1] for x in Wordpron.split() ]
                for i,pronounce in enumerate(sentencePron):
                    if Wordpron[0] == pronounce:  # 该词的第一个拼音和节点一样
                        if len(word) > len(sentencePron[i:]):  # 虽然有拼音一样，但是长度太长了
                            break
                        else:
                            j=i  # 判断后面的几个是否相同
                            while True:
                                if Wordpron[j-i] != sentencePron[j]:  # 很遗憾，后面的有的不相同
                                    break
                                else:
                                    j += 1
                                if j-i >= lenWord:  #此时j已经走出word这个词了
                                    # 这个词可以放到里面去
                                    for k in range(i,i+lenWord):
                                        self.net[k].append(word[k-i])
                                    break
            for i,layer in enumerate(self.net):
                self.net[i] = list( set(layer) )



class Step2(Step1):
    def __init__(self, pronounces: list, lexiconFile: str, ngramsFile: str):
        """
        Parameter(s):
            pronounce: list
                观测值序列，每个拼音为字符串，所有拼音组成一个列表
            lexiconFile: file
                拼音文件        
            ngramsFile:
                语料库
        """
        self.netDict = []

        super().__init__(pronounces, lexiconFile)

        self.ngramsCountsDict = self.ngramCounts(ngramsFile)
        #############################################################
        散文_unigram = dill.load(open("散文_unigram",'rb'))
        散文_bigram = dill.load(open("散文_bigram",'rb'))
        for unigram in 散文_unigram:
            try:
                self.ngramsCountsDict[unigram] += 散文_unigram[unigram]
            except:
                self.ngramsCountsDict[unigram] = 散文_unigram[unigram]

        for bigram in 散文_bigram:
            try:
                self.ngramsCountsDict[bigram] += 散文_bigram[bigram]
            except:
                self.ngramsCountsDict[bigram] = 散文_bigram[bigram]          

        #############################################################


    def ngramCounts(self,file_ngram) -> dict:
        """
        计算unigram的词频
        ----
        Pamater(s):
            file_ngram: file
                ngram语料库
        Return:
            myDict: dict
                每个unigram对应的词频
        """
        if os.path.exists("ngramsCountbDict.dill"):
            myDict = dill.load(open("ngramsCountbDict.dill", 'rb'))
        else:
            with open(file_ngram, encoding='utf-8') as F:
                myDict = {}
                allLines = F.readlines()

                for line in allLines:
                    line = line.split()
                    myDict[line[0]] = int(line[1])
            dill.dump(myDict,open("ngramsCountbDict.dill", 'wb'))
        return myDict


class Step3(Step2):
    def __init__(self, unigramsFile : str, pronounces: list, lexiconFile: str, ngramsFile: str) -> None:
        """
        计算汉字转换拼音的统计概率 

        ----
        Parameter(s):
            unigramsFile: str
                ungram文件名
            pronounces: list
                拼音的输入
            lexiconFile:
                音标文件  lexicon.txt
            ngramsFile:
                124-grams.word这个文件
        """
        self.transmitMatrixT = [ {} for _ in range(len(pronounces))]  # 拼音对应字的概率

        super().__init__(pronounces, lexiconFile,ngramsFile)
        with open(unigramsFile, 'r', encoding='utf-8') as F:
            for line in F:
                word,count = line.split()
                pronouce = pinyin(word, v_to_u=False, style=Style.NORMAL)  # pronounce 是一个二维列表
                for i,pron in enumerate(pronounces):  
                    if pronouce[0][0] == pron:  # 这个字在这个拼音里面
                        self.transmitMatrixT[i][word] = int(count)
        
        total = 0
        for i,pronList in enumerate(self.transmitMatrixT):
            for v in pronList.values():
                total += v
            for key in pronList:
                self.transmitMatrixT[i][key] = pronList[key]/total


class Step4:
    def __init__(self, net, transmitMatrix: list, ngramsCountsDict: dict) -> None:
        """
        每两层之间创建状态转移矩阵

        parameter(s):
            net: list
                网络的节点
            transmitMatrix: list of array
                各层节点的发射概率   （字在音占的比例）
            ngramsCountsDict: dict
                语料库中每个gram对应的出现次数
        """
        self.transformMatixes = []  # 第一个元素放到 第0层那里  这里会顺便加入前向概率和发射概率，所以求的是节点的路径概率
        
        try:
            totalNgrams_count = dill.load(open("totalNgrams_count.dill", 'rb'))
        except:
            totalNgrams_count = 0
            for key in ngramsCountsDict:
                totalNgrams_count += ngramsCountsDict[key]
            dill.dump(totalNgrams_count,open("totalNgrams_count.dill", 'wb'))

        init_proba = np.ones(len(net[0]))/len(net[0])  # 均匀分布
        transmit = []
        for word in net[0]:
            try:  # transmitMatrix是从ungiramFile里面统计出来的，而net是从lexiconFile里面统计出来的，所以net里面的字，transmitMatrix里面可能没有
                transmit.append(transmitMatrix[0][word])  
            except:
                transmit.append(1e-8)

        forward_proba = init_proba * np.array(transmit)

        for k in range(1,len(net)):
            row = len(net[k-1])
            column = len(net[k])
            matrix = np.zeros((column,row))  # 方便矩阵运算，把第二层当作行
            # forward_proba : row x 1
            for i,toNode_str in enumerate(net[k]):
                for j,fromNode_str in enumerate(net[k-1]):
                    conditionalProba = self.getTranformProbability(fromNode_str, toNode_str,ngramsCountsDict,totalNgrams_count)
                    try:
                        transformProba = conditionalProba*forward_proba[j]*transmitMatrix[k][net[k][i]]
                    except:
                        transformProba = conditionalProba*forward_proba[j]*1e-8
                    matrix[i][j] = transformProba

            self.transformMatixes.append(matrix)
            forward_proba = matrix.sum(axis=1)
        

    def getTranformProbability(self,fromNode : str, toNode : str, ngramsCountsDict: dict,totalNgrams_count: int):
        """得到状态转换条件概率"""
        
        """forward_probability里面已经有了P(一）*P(一|yi），所以我们还需要计算P(枝|一）"""
        try:
            doubleWord_count = ngramsCountsDict[fromNode+toNode]  # 我爱 这个词没有出现在ngrams中
            # toNode_count = ngramsCountsDict[toNode]  # 所以用原来的方案 P(枝|一） = P（枝)
        except:
            # toNode_count = 0
            doubleWord_count = 0
        try:
            fromNode_count = ngramsCountsDict[fromNode]
        except:
            fromNode_count = 1  #################################################################################################
        conditionalProba = (1+doubleWord_count)/(fromNode_count+totalNgrams_count)  # 加一平滑

        return   conditionalProba


class Step5:
    def __init__(self, net: list, pathMatrix: list ) -> None:
        """
        使用viterbi算法计算最优路径

        ----
        Parameter(s):
            net: list 
                Step1里面创建的，各个层中都有哪些汉字
            pathMatrix: list of 2d array
                各个节点的路径大小（概率的形式）
        """
        self.path = []

        self.pathMatrix = pathMatrix
        self.net = net
        self.main()
    
    def main(self) -> None:
        """计算最优路径"""
        bestArrows = [ ]  # 记录
        for matrix in self.pathMatrix:
            bestArrow_index = matrix.argmax(axis=1)  # 得到每行的最大值索引
            agent = []
            for i in range(len(matrix)):
                bestArrow = [bestArrow_index[i],i]
                maxProba = matrix[i][bestArrow_index[i]]
                bestArrow.append(maxProba)  # [一，之，0.03]
                agent.append(bestArrow)  # [已，之，0.09]，[一，枝，0.3]
            bestArrows.append(agent)
        
        length = np.zeros(len(bestArrows[-1]))
        for i,path in enumerate(bestArrows[-1]):
            arrowLength = -np.log(path[2])
            fromNode = path[0]
            for j in range(len(bestArrows)-2, -1, -1):  # 从倒数第二层开始，因为前面已经算了arrowLength
                for arrow in bestArrows[j]:
                    if arrow[1] == fromNode:  # 这个arrow指向了j层的fromNode
                        arrowLength += -np.log(arrow[2])
                        fromNode = arrow[0]
            length[i] = arrowLength
        bestPath_index = length.argmin()

        toNode = bestArrows[-1][bestPath_index][1]
        fromNode = bestArrows[-1][bestPath_index][0]
        toWord = self.net[-1][toNode]
        fromWord = self.net[-2][fromNode]
        self.path.append( (toWord,fromWord) )
        for i,arrows in enumerate(bestArrows[-2::-1]):
            for arrow in arrows:
                if arrow[1] == fromNode:
                    fromNode = arrow[0]
                    toWord = fromWord
                    fromWord = self.net[len(self.net)-3-i][fromNode]
                    self.path.append( (toWord,fromWord) )
                    
                    break

def display(path :list):
    """
    parameter(s):
        path: list of list
            记录了最优路径，每个元素为 [ 当前节点的名字（字），上一层节点的名字（字）]
    """
    result = ''
    for node in path[::-1]:
        result += node[1]
    result += node[0]
    return result


if __name__ == "__main__":
    lexicon = "lexicon.txt"
    ngramsFile = "人民日报96年语料.124-gram.word"
    unigramsFile = "人民日报96年语料.1-gram.word"
    pronounce = 'sentence'
    while pronounce:
        pronounce = input("请输入拼音，使用空格分隔(注意：略的拼音为：lu:e )：\n").split()
        # pronounce = "ba ji si tan"
        S3 = Step3(unigramsFile,pronounce,lexicon,ngramsFile)
        for lay in S3.net:
            for word in lay[:5]:
                print(word)
            print()
        S4 = Step4(S3.net,S3.transmitMatrixT,S3.ngramsCountsDict)
        S5 = Step5(S3.net,S4.transformMatixes)
        result = display(S5.path)
        print(result)