# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:11:52 2019

@author: Black_Death
"""

#updating github

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Perceptron:
    
    def relu_activation_function(self,x):
        if x < 0:
            return 0
        return x
    
    def step_activation_function(self,x):
        if x < 0:
            return 0
        else:
            return 1
        
    def label_encoder(self,labels):
        unique_labels = {}
        encode_labels = [0] * len(labels)
        
        encode = 0
        for index,label in enumerate(labels):
           if label not in unique_labels:
               unique_labels[label] = encode
               encode +=1
               
           encode_labels[index] = unique_labels[label]
       
        return encode_labels

    def update_weights(self,weights,change_in_weights):
        return np.add(weights,change_in_weights)
    
    def output_weights(self):
        weightsFile = open('weights.txt','w')
        for index,weight in enumerate(self.weights):
            weightsFile.write('{0}\n'.format(weight))
        
        weightsFile.close()

    def normalize_feature(self,feature):
        min_value = np.min(feature)
        max_value = np.max(feature)     
        return list(map(lambda x:(x-min_value)/(max_value-min_value),feature))

    def calculate_error(self,output,prediction):
    #    sum_of_square_error = 0
    #    
    #    for i in range(0,len(prediction)):
    #        sum_of_square_error += (prediction[i] - outputs[i])**2
    #    
    #    average_of_error = sum_of_square_error/(2*len(prediction))
    #    return average_of_error
        
        return output - prediction
     
    def initialize_weights(self,number_of_weights):
        weights = [0]* number_of_weights
        
        for index,weight in enumerate(weights):
            weights[index] = round(random.random(),4)
        
        return weights

    def get_confusion_matrix(self,predictions,outputs):
        matrix = [[0,0],[0,0]]
        for i in range(0,len(predictions)):
            matrix[predictions[i]][outputs[i]] += 1
        return matrix
    
    def output_final_results(self,predictions,outputs,file,progress_bar=None):
        
        cm = self.get_confusion_matrix(predictions,outputs)
        accuracy = ((cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1]  + cm[1][0] + cm[0][1]))*100
        precision = ((cm[0][0]/(cm[0][0] + cm[0][1] + 1)))*100
        recall = ((cm[0][0])/(cm[0][0] + cm[1][0] + 1))*100
        #print("Accuracy : {0}".format(accuracy))
        #print("Precision : {0}".format(precision))  
        #print("Recall : {0}".format(recall))
        
        file.write("Accuracy  : {0:<10.4f}\n".format(accuracy,4))
        file.write("Precision :{0:<10.4f}\n".format(precision,4))
        file.write("Recall    :{0:<10.4f}\n".format(recall,4))
        
        if progress_bar is not None:
            progress_bar.set_postfix({"Accuracy":accuracy,'Precision':precision,"Recall":recall})        
    
    
    def output_metrics_results(self,predictions,outputs,file):
        cm = self.get_confusion_matrix(predictions,outputs)
        accuracy = ((cm[0][0] + cm[1][1])/(cm[0][0] + cm[1][1]  + cm[1][0] + cm[0][1]))*100
        precision = ((cm[0][0]/(cm[0][0] + cm[0][1] + 1)))*100
        recall = ((cm[0][0])/(cm[0][0] + cm[1][0] + 1))*100
#        print("Accuracy : {0}".format(accuracy))
#        print("Precision : {0}".format(precision))  
#        print("Recall : {0}".format(recall))
        
        file.write("{0:<10.4f}".format(accuracy))
        file.write("{0:<10.4f}".format(precision))
        file.write("{0:<10.4f}\n".format(recall))
        

    
    def iteration_results(self,weights,prediction,output,file):

        for i in range(len(weights)):
            file.write("{0:<10.4f}".format(weights[i]))
        
        file.write("{0:<10}".format(output))
        file.write("{0:<10}".format(prediction))

    def fit(self,X,y,learning_rate=0.1,epochs=100):
        
        self.X_train = X
        self.y_train = y

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        self.weights = self.initialize_weights(X.shape[1] + 1)
        
        epochs_data = []
        
        resultsFile = open('train_results.txt','w')

        progress_bar = tqdm(range(0,epochs),position=0, leave=True)
        for epoch in progress_bar:
            
            learner_check = True
            #print("========================== Epoch : {0} Starts ==========================".format(epoch + 1))
            resultsFile.write("========================== Epoch : {0} Starts ==========================\n\n\n".format(epoch + 1))

            predictions = []
            
            for index,instance in enumerate(X_train):

                error = 0
                
                instance = np.append(instance,-1)
                prediction = self.step_activation_function(np.dot(instance,self.weights)) 

                if prediction != y_train[index]:
                    learner_check = False

                    error = self.calculate_error(y_train[index],prediction)
                    change_in_weight = np.multiply(instance,learning_rate * error) 
                    self.weights = self.update_weights(self.weights,change_in_weight)          
                    
                self.iteration_results(self.weights,prediction,y_train[index],resultsFile) 
                resultsFile.write("{0:<10}\n".format(round(error,4)))

                #self.output_metrics_results(val_predictions,y_val,resultsFile)
 
            predictions.append(prediction)
            epochs_data.append(predictions)       
            
            
            #print("========================== Epoch : {0} Results ==========================")
            resultsFile.write("========================== Epoch : {0} Results ==========================\n".format(epoch + 1))
            
            val_predictions = self.predict(X_val)   
            self.output_final_results(val_predictions,y_val,resultsFile,progress_bar)
            
            #print("========================== Epoch : {0} Ends ==========================".format(epoch))
            resultsFile.write("\n========================== Epoch : {0} Ends ==========================\n\n\n".format(epoch + 1))
            
            if(learner_check): 
                break


        resultsFile.close()
        self.output_weights()
            
    def predict(self,X):
        
        self.X_test = X
        
        predictions = []
        for instance in X:
            instance = np.append(instance,-1)
            prediction = self.step_activation_function(np.dot(instance,self.weights))
            predictions.append(prediction)
        return predictions



import sys

if '--Learning' in sys.argv:
    
    #Importing The Dataset

    dataset = pd.read_csv(r'{0}'.format(sys.argv[1]),header=None)
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    #Making Perceptron Model And Preprocessing Data
    
    perceptron = Perceptron()
    X[:,0] = perceptron.normalize_feature(X[:,0])
    X[:,1] = perceptron.normalize_feature(X[:,1])
    X[:,2] = perceptron.normalize_feature(X[:,2])
    X[:,3] = perceptron.normalize_feature(X[:,3])
    X[:,4] = perceptron.normalize_feature(X[:,4])
    X[:,5] = perceptron.normalize_feature(X[:,5]) 
    X[:,6] = perceptron.normalize_feature(X[:,6])
    X[:,7] = perceptron.normalize_feature(X[:,7])
    perceptron.fit(X,y,epochs=5000,learning_rate=0.01)

elif '--Test' in sys.argv:
    #Importing The Dataset

    dataset = pd.read_csv(r'{0}'.format(sys.argv[1]))
    X = dataset.iloc[:,:-1].values
    y = dataset.iloc[:,-1].values

    #Making Perceptron Model And Preprocessing Data
    
    perceptron = Perceptron()
    X[:,0] = perceptron.normalize_feature(X[:,0])
    X[:,1] = perceptron.normalize_feature(X[:,1])
    X[:,2] = perceptron.normalize_feature(X[:,2])
    X[:,3] = perceptron.normalize_feature(X[:,3])
    X[:,4] = perceptron.normalize_feature(X[:,4])
    X[:,5] = perceptron.normalize_feature(X[:,5]) 
    X[:,6] = perceptron.normalize_feature(X[:,6])
    X[:,7] = perceptron.normalize_feature(X[:,7])
    
    weights = []
    
    weightsFile = open('weights.txt','r')
    for line in weightsFile.readlines():
        weights.append(float(line))
    
    perceptron.weights = weights
    
    results_file = open('test_results.txt','w')
    
    predictions = perceptron.predict(X)
    perceptron.output_final_results(predictions,y,results_file)
    
    results_file.close()
     
else :
    print("Invalid Command Line Arugments. Please Try Again :)")





