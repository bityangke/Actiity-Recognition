"""

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                         Long-Short Term Module                           |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright 2016-2020, Marcos Vinicius Teixeira               |
//|                          All Rights Reserved.                            |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
//  OVERVIEW: LSTM.py
//  ========
//
//
"""

from __future__ import print_function

# !/usr/bin/env python
# LSTM

import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.validation import testOnSequenceData
from pybrain.structure.modules import LSTMLayer, SoftmaxLayer
from pybrain.datasets import SequenceClassificationDataSet
from pybrain.supervised import RPropMinusTrainer, BackpropTrainer
from os.path import exists
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader



# np.set_printoptions(threshold=np.nan)

class LSTM:
    """ LSTM layer"""

    def __init__(self):

        self.X =  []
        self.y =  []
        self.DS = {}

        # metric vars
        self.precision =[]
        self.recall =   []
        self.f1 =       []
        self.accuracy = []

    def fit(self,X_train,y_train):
        # Creating training and test data
        self.X = X_train
        self.y = y_train

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


        # SequenceClassificationDataset(inp,target, nb_classes)
        # inp = input dimension
        # target = number of targets
        # nb_classes = number of classes
        trndata = SequenceClassificationDataSet(100, 1)

        for index in range(len(y_train)):
            trndata.addSample(X_train[index], y_train[index])

        trndata._convertToOneOfMany(bounds=[0., 1.])

        if exists("params.xml"):
            self.rnn = NetworkReader.readFrom('params.xml')
        else:
            # construct LSTM network - note the missing output bias
            self.rnn = buildNetwork(trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer,
                               outputbias=False, recurrent=True)

        # define a training method
        self.trainer = BackpropTrainer(self.rnn, dataset=trndata, momentum=0.9, learningrate=0.00001,verbose=False)

        step=0
        # lets training (exclamation point)
        for i in range(50):
            # setting the ephocs for the training
            self.trainer.train()
            print(step)
            step+=1
            # calculating the error
            trnresult = 100. * (1.0 - testOnSequenceData(self.rnn, trndata))
            #print("train error: %5.2f%%" % trnresult)

            # activating the softmax layer
            out = self.rnn.activate(X_train[0])
            out = out.argmax(axis=0)

        # saving the params
        NetworkWriter.writeToFile(self.rnn, 'params.xml')
        
    def predict(self,X_test):

        # evaluate the net in test data
        result = []
        final_result=[]

        for x in X_test:
            result.append(self.rnn.activate(x).argmax())

        return result


