# Perceptron-in-Python
Perceptron implementation in python for Iris dataset

In this example I have taken Iris dataset to train 2 class identifier.  Iris data set is 3 class data set. But you can use it as 2 class data set by removing data for iris-virginica. Here Iris.setosa and Iris.versicolor data can act as 2 class data set as they can be easily separated by boundary with respect to attribute value [sepal.length, sepal.width, petal.length, petal.width]

# Explanation of different training dataset

1. Training dataset 4: small size training dataset. 17 records to training.
2. Training dataset 3: medium size training dataset. 40 records to training. 
3. Training dataset 1: large size training dataset. 76 records to training. 
4. Training dataset 2:  26 records. No sorted-on basis of prediction.
5. Test dataset: 100 records.  
Here these three datasets have record in ascending order with respect to prediction. Where first half records are with prediction 0 (Iris.seton) and second half are with prediction1 (Iris.versicolor)



# Conclusion 

Once perceptron is trained I tested it with my test data. It can accuratlly predict class for flowers. Here I tried to identify effect of winsorizing for training perceotron and accuracy once its trained. In this case effect depends on dataset I use for training perceptron. It may be different for different dataset. Overall when I used winsorized data, it reduced training time and also improved accuracy for test data.

You can use this perceptron for any two class dataset. I tested this with Sonar dataset.

Credits: To build this perceptron I refered https://machinelearningmastery.com/. I want to give creadit to Dr. Jason Brownlee for providing amazing materials.

# References 

https://en.wikipedia.org/wiki/Winsorizing
https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/
https://en.wikipedia.org/wiki/Perceptron
https://en.wikipedia.org/wiki/Iris_flower_data_set
https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
