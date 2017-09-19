#Program for preceptron on two class dataset of iris
#This program used Iris dataset, from Iris  dataset only values for Iris setosa and Iris Versicolor is taken to train this preceptron
#Dataset modified here for Iris setoda prediction value will be 0 and for Iris versicolor prediction value will be 1

import scipy.stats
import numpy as npy


# 'dataset' is used to train perceptron
dataset = npy.genfromtxt('traindata4.csv', delimiter=',')
print("Execution Start:")
print("Trainig dataset:")
print(dataset)
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')


# 'winzdataset' is updated in future after winsorizing values of 'dataset'
winzdataset = npy.genfromtxt('traindata4.csv', delimiter=',')

# 'testdataset' contains data for testingn of perceptron once they are trained
testdataset =  npy.genfromtxt('testdata.csv', delimiter=',')

#column array to store imput values for winsorizing , each array here represents column from 'dataset'
clm1 = []
clm2 = []
clm3 = []
clm4 = []

#store each column into arrrays
for ary in dataset:
    clm1.append(ary[0])
    clm2.append(ary[1])
    clm3.append(ary[2])
    clm4.append(ary[3])

#winsorizing each array
cl1 = scipy.stats.mstats.winsorize(clm1, limits=0.1)
cl2 = scipy.stats.mstats.winsorize(clm2, limits=0.1)
cl3 = scipy.stats.mstats.winsorize(clm3, limits=0.1)
cl4 = scipy.stats.mstats.winsorize(clm4, limits=0.1)

#store winsorized data into winzdataset
i = 0
for ary in winzdataset:
    ary[0]=cl1[i]
    ary[1]=cl2[i]
    ary[2]=cl3[i]
    ary[3]=cl4[i]
    i = i +1
print("Winsorized trainig dataset:")
print(winzdataset)
# Make a prediction with weights
def predict_class(record, weights):
    activation = weights[0]
    for i in range(len(record) - 1):
        activation += weights[i + 1] * record[i]
    return 1.0 if activation >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def train_preceptron(records, learning_rate, n_iteraton):
    weights = [0.0 for i in range(len(records[0]))]
    for iteration in range(n_iteraton):
        sum_error = 0.0
        for row in records:
            prediction = predict_class(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('Iteration no=%d, learning rate=%.3f, errors in iteration=%d' % (n_iteraton, learning_rate, sum_error))
    return weights


# test trained preceptrop for  test data
def test_preceptron(dataset,weights):
    sumer = 0.0
    for row in dataset:
        prediction = predict_class(row, weights)
        print('Class = %d, Prediction = %d '%(row[-1],  prediction))
        error = row[-1] - prediction
        sumer += error ** 2
    print(
        '-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')

    print("No of wrong prediction : =%d " %(sumer))


l_rate = 0.05
n_epoch = 7
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print("Train perceptron with original dataset:")
weights = train_preceptron(dataset, l_rate, n_epoch)
print("final weight for perceptron:")
print(weights)
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print("Test perceptron with test dataset")
toterror = test_preceptron(testdataset,weights)
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print("Train perceptron with winsorized dataset:")
weights = train_preceptron(winzdataset, l_rate, n_epoch)
print("final weight for perceptron:")
print(weights)
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print("Test perceptron with test dataset")
toterror = test_preceptron(testdataset,weights)
print('-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print("Execution ends:")
# end of program

