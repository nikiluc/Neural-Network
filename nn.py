import numpy as np
from scipy.io import arff
import pandas as pd
from scipy import stats
from random import randrange
from random import seed
import time
import sys


sys.stdout = open('adults.out', 'a')

np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
pd.set_option('display.max_columns', None)

seed(1)

file = 'adult-big.arff'

data = arff.loadarff(file)

df = pd.DataFrame(data[0]) # loading file as a dataframe

convert = df.select_dtypes([np.object])
convert = convert.stack().str.decode('utf-8').unstack() # remove b

for col in convert:
    df[col] = convert[col]


df.replace('?', np.NaN, inplace = True) # replace all question marks with NAN

df = df.apply(lambda x:x.fillna(x.mode().iloc[0])) #fill missing values

dummy_wc = pd.get_dummies(df['workclass'])
dummy_edu = pd.get_dummies(df['education'])
dummy_ms = pd.get_dummies(df['marital-status'])
dummy_oc = pd.get_dummies(df['occupation'])
dummy_re = pd.get_dummies(df['relationship'])
dummy_ra = pd.get_dummies(df['race'])
dummy_sex = pd.get_dummies(df['sex'])
dummy_nc = pd.get_dummies(df['native-country'])
dummy_cl = pd.get_dummies(df['class'])

df = df.drop(['workclass', 'education', 'marital-status', 'occupation',
              'relationship', 'race', 'sex', 'native-country', 'class'], axis=1)

#dummy encoding
df = pd.concat([df, dummy_wc, dummy_edu, dummy_ms, dummy_oc, dummy_re, dummy_ra, dummy_sex, dummy_nc, dummy_cl], axis=1)

#numpy array
test = df.values


#z-score normalization

#age
test[0:][:, 0] = stats.zscore(test[0:][:, 0])

#fnlwgt
test[0:][:, 1] = stats.zscore(test[0:][:, 1])

#education-num
test[0:][:, 2] = stats.zscore(test[0:][:, 2])

#capital-gain
test[0:][:, 3] = stats.zscore(test[0:][:, 3])

#capital-loss
test[0:][:, 4] = stats.zscore(test[0:][:, 4])

#hours-per-week
test[0:][:, 5] = stats.zscore(test[0:][:, 5])



# stores results from each model

class Neural_Network:

    def __init__(self):

        self.inputSize = 105
        self.outputSize = 2
        self.hiddenSize = 60

        #initialize weights and biases

        self.w = np.random.uniform(-0.5, 0.5, (self.inputSize, self.hiddenSize))

        self.w2 = np.random.uniform(-0.5, 0.5, (self.hiddenSize, self.outputSize))

        self.b = np.random.uniform(-0.5, 0.5, self.hiddenSize)
        self.b2 = np.random.uniform(-0.5, 0.5, self.outputSize)

    def feedforward(self, x):

        self.p = np.dot(x, self.w) #input times weight
        self.p1 = self.p + self.b # + bias

        self.p2 = self.sigmoid(self.p1) # sigmoid

        self.q = np.dot(self.p2, self.w2) #hidden x weight
        self.q1 = np.add(self.q, self.b2) # + bias
        output = self.sigmoid(self.q1) # sigmoid

        return output

    def sigmoid(self, sm):

        return 1 / (1 + np.exp(-sm))

    def derivative_sigmoid(self, d):

        dp = 1 / (1 + np.exp(-d))

        return dp * (1 - dp)

    def backpropagation(self, x, y, output, lr):

        # correct minus output

        self.error = y - output

        self.e_final = self.error * self.derivative_sigmoid(output)  #sigmoid derivative

        self.s_error = self.e_final.dot(self.w2.T)  # second error - hidden weights error

        self.p2_final = self.s_error * self.derivative_sigmoid(self.p2)  # sigmoid derivative

        #calculations to update weights and biases

        wi = x.T.dot(self.p2_final)
        wi2 = self.p2.T.dot(self.e_final)

        wi *= lr # multiply by learning rate
        wi2 *= lr

        self.w += wi  # input to hidden weights
        self.w2 += wi2  # hidden to output weights

        b = lr * np.mean(self.p2_final, axis=0) # updating array of biases set 1
        b2 = lr * np.mean(self.e_final, axis=0) # updating array of biases set 2

        self.b += b
        self.b2 += b2

        #calculate two output node errors

        node1 = 0
        node2 = 0

        i = 0
        # averaging all of the output errors
        for item in self.e_final:
            i += 1
            node1 += item[0]
            node2 += item[1]

        node1 = node1/i
        node2 = node2/i

        errors = np.array([node1, node2])

        return errors


    #trains the model
    def train(self, split, lr):

        error = 0

        for fold in split:


            split_data = fold[0:][0:]

            split_data = np.array(split_data)

            split_ans = split_data[0:][:, [105, 106]]

            split_data = split_data[0:][:, 0:105]


            output = self.feedforward(split_data)
            error += self.backpropagation(split_data, split_ans, output, lr) #returns error avg from 4 folds

        error = error/4

        return error


    def predict(self, testdata):

        # stores results from each model

        mipavg = []

        miravg = []

        mapavg = []

        maravg = []

        mif1 = []

        maf1 = []

        accavg = []

        countdata = {}  # dictionary for confusion matrix values for each label
        # List per label
        tpList = []
        fpList = []
        tnList = []
        fnList = []
        recalls = []
        precis = []
        f1List = []

        # confusion matrix values for each label
        countdata["[0.000000 1.000000]trueP"] = 0
        countdata["[1.000000 0.000000]trueP"] = 0
        countdata["[0.000000 1.000000]falseP"] = 0
        countdata["[1.000000 0.000000]falseP"] = 0
        countdata["[1.000000 0.000000]trueN"] = 0
        countdata["[0.000000 1.000000]trueN"] = 0
        countdata["[0.000000 1.000000]falseN"] = 0
        countdata["[1.000000 0.000000]falseN"] = 0

        classes = ['[1.000000 0.000000]', '[0.000000 1.000000]']
        #<=50K and >50K

        for item in testdata:

            data = item[0:105]


            sol = item[105:107]

            prediction = self.feedforward(data)

            if prediction[0] > prediction[1]:
                prediction = np.array([1.000000, 0.000000])

            else:
                prediction = np.array([0.000000, 1.000000])

            if np.array_equal(prediction, sol):

                countdata[str(prediction) + "trueP"] += 1

                for label in classes:

                    if label != str(sol):
                        countdata[str(label) + "trueN"] += 1  # right answer but for different class label

            else:
                # incorrect prediction
                countdata[str(sol) + "falseN"] += 1

                countdata[str(prediction) + "falseP"] += 1


        for label in classes:
            tpList.append(countdata[str(label) + "trueP"])

            tnList.append(countdata[str(label) + "trueN"])

            fpList.append(countdata[str(label) + "falseP"])

            fnList.append(countdata[str(label) + "falseN"])

            p = self.precision(countdata[str(label) + "trueP"], countdata[str(label) + "falseP"])

            r = self.calcRecall(countdata[str(label) + "trueP"], countdata[str(label) + "falseN"])

            precis.append(p)

            recalls.append(r)

            f1List.append(self.calcf1(r, p))

            # calculations of micro/macro precision, recall, f1 and accuracy

        microPrec = self.microPrecision(tpList, fpList)

        microRec = self.microRecall(tpList, fnList)

        macroPrec = self.calcMacro(precis)

        macroRec = self.calcMacro(recalls)

        microF1 = self.calcf1(microRec, microPrec)

        macroF1 = self.calcMacro(f1List)

        accuracy = self.calcAccuracy(tpList, tnList, fpList, fnList)

        # store values from each iteration

        mipavg.append(microPrec)

        miravg.append(microRec)

        mapavg.append(macroPrec)

        maravg.append(macroRec)

        mif1.append(microF1)

        maf1.append(macroF1)

        accavg.append(accuracy)

        finalmicroP = NN.calcAverage(mipavg)
        finalmicroR = NN.calcAverage(miravg)
        finalmacroP = NN.calcAverage(mapavg)
        finalmacroR = NN.calcAverage(maravg)
        finalmacroF1 = NN.calcAverage(maf1)
        finalmicroF1 = NN.calcAverage(mif1)
        finalaccuracy = NN.calcAverage(accavg)

        print("Results: ")

        print("Average Micro Precision: " + str(finalmicroP))

        print("Average Micro Recall: " + str(finalmicroR))

        print("Average Micro F1: " + str(finalmicroF1))

        print("Average Macro Precision: " + str(finalmacroP))

        print("Average Macro Recall: " + str(finalmacroR))

        print("Average Macro F1: " + str(finalmacroF1))

        print("Average Accuracy: " + str(finalaccuracy))


    def precision(self, tp, fp):  # calculate precision
        return (tp) / (tp + fp)

    def calcRecall(self, tp, fn):  # calculate recall
        return (tp) / (tp + fn)

    def calcf1(self, r, p):  # calculate f1

        return (2 * r * p) / (r + p)

    def microRecall(self, tpList, fn):  # calculate micro recall

        return sum(tpList) / (sum(tpList) + sum(fn))

    def microPrecision(self, tpList, fpList):  # calculate micro precision

        return (sum(tpList)) / ((sum(tpList)) + sum(fpList))

    def calcMacro(self, items):  # for both macro precision and recall

        return (sum(items)) / (len(items))

    def calcAccuracy(self, tpList, tnList, fpList, fnList):  # calculate accuracy

        return (sum(tpList) + sum(tnList)) / (sum(tpList) + sum(tnList) + sum(fpList) + sum(fnList))

    def calcAverage(self, list):

        return (sum(list)) / (len(list))


NN = Neural_Network()
np.random.shuffle(test)
j = 0
folds = 5

# Implementation of 5-fold Cross Validation

copy = list(test)
split = list()

sizeofFold = int(len(test) / folds)

# split the data into folds

for i in range(folds):

    fold = list()

    while len(fold) < sizeofFold:
        index = randrange(len(copy))

        fold.append(copy.pop(index))

    split.append(fold)

train = None

model = 0

while j < folds:

    model += 1
    print('Model #' + str(model))

    lr = 0.3 # learning rate

    print('Learning Rate: ' + str(lr))

    print('Hidden Layer Nodes: ' + str(NN.hiddenSize))

    # re-append previously popped fold for new model
    if train != None:
        split.append(train)

    testdata = split[0]

    testdata = np.array(testdata)

    train = split.pop(0)

    iterations = 100

    print("Iterations: " + str(iterations))

    start = time.time() #record time

    for i in range(iterations):  # trains the NN ___ times

        error_val = NN.train(split, lr) #trains neural network

        if i == 0:
            print("Error (First Epoch): " + str(error_val))
        elif i == iterations - 1:
            print("Error (Last Epoch): " + str(error_val))


    end = time.time()
    print("Train time: " + str(end - start))

    NN.predict(testdata) #testing

    # stores results from each model


    j = j + 1

