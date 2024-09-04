import numpy as np
import math
from Preprocessing import data_extraction
from Preprocessing import load_data
from Preprocessing import preocessing_data




class LogisticRegression():
    """
    This class does logistic regression. The weight and biases starts as zero
    and gets adjusted accoringly. 
    """
    
    def __init__(self,lr,epochs):

        self.epochs = epochs
        self.lr = lr
        self.weights = 0
        self.bias = 0
        

    def fit(self,X,Y):
        """
        This function adjust and fits the weight and biases to fit the data,
        using gradient decent, in order to use gradient decent we use the derivative
        of the loss function in respect to W and B. 

        input: data
        return: updates weight(s) and bias
        """
        n,features = X.shape
        self.weights =  np.zeros(features,dtype=np.float64)
            #makes the weights a matrix with the 
            #same lenght as amount of features.

        for _ in range(self.epochs):

            logistic_hat = self.prediction(X)
            error = logistic_hat - Y
            der_weight = -(2/n) * np.dot(X.T,error)
            der_bias = -(2/n) * np.sum(error)
                #derviatives in respect to weight and bias, 
                # full explaination in report.
            
            der_weight = np.asarray(der_weight, dtype=np.float64)
                #had some type issues, this line fixed it. 

            self.weights +=  self.lr * der_weight
            self.bias += self.lr * der_bias
                #updates the weight(s) and bias usings the 
                #opposite direction of the derivative


        return self.weights,self.bias

 
    def prediction(self,X):
        """
        This function makes the predictions both linear and logistic
        """

        linear_hat = np.dot(X,self.weights) + self.bias
            #Since X is a matrix, it's neccesary to do matrix multiplication. 
        logistic_hat = self.sigmoid(linear_hat)
            #pass the W * x + b into the x variable of the function.
        
        return logistic_hat
    

    def sigmoid(self,x): 
        """
        This functions caps the range [0,1].
        """

        x = np.asarray(x, dtype=np.float64)
            #got typeerror, this line fixed it

        return 1 / (1 + np.exp(-x))


    def evalute(self,X_test,Y_test):
        """
        This function uses the current weight(s) and bias and 
        makes prediction using these values. It then compares
        the machine predictions to the labels of the test set.

        input: test data set
        return: accuracy, confusion matrix
        """

        n,features = X_test.shape
        predictions = []
                        
        logistic_hat = self.prediction(X_test)

        for numbers in logistic_hat:
            if numbers <= 0.5:
                predictions.append(0)
            else:
                predictions.append(1)

        predictions = np.array(predictions)
            #convert into array

        equal_elements = predictions == Y_test
        successful_guesses = np.sum(equal_elements)
            #Count how many times they have equal numbers at equal indices
            #chatgpt helped me with this. 

        confusion_matrix = (self.confusion_matrix(predictions,Y_test))


        accuracy = successful_guesses/n

        return round(accuracy,3),confusion_matrix
    

    def confusion_matrix(self,predictions,Y_test):
        """
        returns the amount of true positive,false positive,true negative and false negative.
        This function is used withing the evaluate function, therefore this is just a 
        helping function.
        This is made as a seperate function to make the code more readable.

        input: predictions, Y_test

        returns: 
             array of [[TN,FP],[FN,TP]] 
        """

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        
        for index in range(len(predictions)):
            if predictions[index] == 1:
                if Y_test[index] == 1:
                    TP += 1
                else:
                    FP += 1
            if predictions[index] == 0:
                if Y_test[index] == 0:
                    TN += 1
                else:
                    FN += 1

        confusion_matrix = np.array([[TN, FP],
                                     [FN, TP]])
        return confusion_matrix




if __name__ == '__main__':

    """
    1a
    """

    headers,data_array = load_data('SpotifyFeatures.csv')
    print("\n1a:")
    print(f"There is {data_array.shape[0]} songs, and {data_array.shape[1]} features in the dataset.")

    """
    1b
    """

    A,B,header_label = data_extraction(data_array,headers)
    print("\n1b:")
    print(f"Pop data: {A.shape},Classical data: {B.shape}")

    """
    1c
    """

    X_train, X_test, y_train, y_test =(preocessing_data(A,B))

    print("\n1c:")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(y_train, dtype=np.float64)

    """
    2a
    """

    lr = LogisticRegression(0.05,250)
    weights,bias =((lr.fit(X_train,y_train)))
    print("\n2a:")
    print(f"Weights:{weights},bias:{bias}")
    accuracy,confusion_matrix = lr.evalute(X_test,y_test)
    print(f"accuracy:{accuracy}")

    """
    3a
    """

    print("\n3a:")
    print(f"confusion matrix:\n {confusion_matrix}")