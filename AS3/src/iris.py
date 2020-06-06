# -*- coding: utf-8 -*-

#Importing Packages and Lib
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras import models
from keras import layers
from keras import optimizers
from keras.models import load_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout




#Loading the data
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data'
iris_data = pd.read_csv(url, error_bad_lines=False,header=None)

#Extracting predictors and response variables 
features = iris_data.iloc[:,:-1]
label = iris_data.iloc[:,-1:]

#One Hot encoding of the response variable 
encoder = OneHotEncoder()
label = np.reshape(label, (label.shape[0], 1))
label= encoder.fit_transform(label).toarray()

#Spliting of the data into train and test data. 
x_train,x_test,y_train,y_test = train_test_split(features, label,test_size = 0.2)

#Normalizing the data
train_mean = x_train.mean()
train_std = x_train.std()
x_train = ((x_train - train_mean )/train_std)
x_test = ((x_test - train_mean)/train_std)

#Spliting data into Train and Validation data.
x_train,x_val,y_train,y_val = train_test_split(x_train, y_train,test_size = 0.2)


def loss_accuracy_plot(history_dict,name):  
    plt.clf()
    
    #Plotting the Loss values 
    loss_values=history_dict['loss']
    val_loss_values=history_dict['val_loss']
    epochs=range(1,len(loss_values)+1)
    plt.plot(epochs,loss_values,'r',label='Training Loss')
    plt.plot(epochs,val_loss_values,'b',label='Validation Loss')
    plt.title(name+': Traning and Validation Loss ')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    #Clear Previous Plots
    plt.clf()
    
    #Plot the Accuracy values 
    acc_values=history_dict['accuracy']
    val_acc_values=history_dict['val_accuracy']
    plt.plot(epochs,acc_values,'r',label='Training Accuracy')
    plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
    plt.title(name+' :Traning and Validation Accuracy ')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()



# Evaluating Diffrent Loss Functiuons

#Categorical_Crossentropy
model_cat_cross_entropy = models.Sequential()
model_cat_cross_entropy.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_cat_cross_entropy.add(layers.Dense(16,activation='relu'))
model_cat_cross_entropy.add(layers.Dense(3,activation='softmax'))
rmsprop = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model_cat_cross_entropy.compile(optimizer=rmsprop,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#60
history_cat_cross_entropy=model_cat_cross_entropy.fit(x_train,y_train,epochs=60,batch_size=8,validation_data=(x_val,y_val))
history_dict_cat_cross_entropy=history_cat_cross_entropy.history
model_cat_cross_entropy.save("cat_cross_entropy.hdf5")
loss_accuracy_plot(history_dict_cat_cross_entropy,name='Categorical Cross Entropy Loss')

#Loading the saved Model with categorical crossentropy 
model_new_cat_cross_entropy=load_model("cat_cross_entropy.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_cat_cross_entropy = model_new_cat_cross_entropy.evaluate(x_test,y_test)
print('Loss: {}'.format(result_cat_cross_entropy[0]))
print('Accuracy: {}'.format(result_cat_cross_entropy[1]))




#Kullback_leibler_divergence Loss

model_KL=models.Sequential()
model_KL.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_KL.add(layers.Dense(16,activation='relu'))
model_KL.add(layers.Dense(3,activation='softmax'))
rmsprop_KL=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model_KL.compile(optimizer=rmsprop_KL,loss=losses.kullback_leibler_divergence,metrics=['accuracy'])
#80
history_KL=model_KL.fit(x_train,y_train,epochs=50,batch_size=4,validation_data=(x_val,y_val))
history_dict_KL=history_KL.history
model_KL.save("KL_Divergence.hdf5")
loss_accuracy_plot(history_dict_KL,name='KL Divergence Loss')

#Loading saved Models
model_new_KL=load_model("KL_Divergence.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_KL=model_new_KL.evaluate(x_test,y_test)
print('Loss: {}'.format(result_KL[0]))
print('Accuracy: {}'.format(result_KL[1]))





#Hinge Losses
model_hinge = models.Sequential()
model_hinge.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_hinge.add(layers.Dense(16,activation='relu'))
model_hinge.add(layers.Dense(16,activation='relu'))
model_hinge.add(layers.Dense(3,activation='softmax'))
rmsprop_hinge=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model_hinge.compile(optimizer=rmsprop_hinge,loss=losses.hinge,metrics=['accuracy'])
#80
history_hinge=model_hinge.fit(x_train,y_train,epochs=50,batch_size=4,validation_data=(x_val,y_val))
history_dict_hinge=history_hinge.history
model_hinge.save("hinge_loss.hdf5")
loss_accuracy_plot(history_dict_hinge,name='Hinge Loss')

#Loading saved Models
model_new_hinge=load_model("hinge_loss.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_hinge=model_new_hinge.evaluate(x_test,y_test)
print('Loss: {}'.format(result_hinge[0]))
print('Accuracy: {}'.format(result_hinge[1]))



#Squared Hinge Loss
model_squared_loss=models.Sequential()
model_squared_loss.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_squared_loss.add(layers.Dense(16,activation='relu'))
model_squared_loss.add(layers.Dense(3,activation='softmax'))
rmsprop_squared_loss=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model_squared_loss.compile(optimizer=rmsprop_squared_loss,loss=losses.squared_hinge,metrics=['accuracy'])
#60
history_squared_loss=model_squared_loss.fit(x_train,y_train,epochs=60,batch_size=4,validation_data=(x_val,y_val))
history_dict_squared_loss=history_squared_loss.history
model_squared_loss.save("sq_hinge_loss.hdf5")
loss_accuracy_plot(history_dict_squared_loss,name='Squared Hinge Loss')

model_new_squared_loss=load_model("sq_hinge_loss.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_squared_loss=model_new_squared_loss.evaluate(x_test,y_test)
print('Loss: {}'.format(result_squared_loss[0]))
print('Accuracy: {}'.format(result_squared_loss[1]))





#Evaluating Diffrent Optimizers on data


#SGD Optimizers
#---------------------------------------------------------------------
model_SGD=models.Sequential()
model_SGD.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_SGD.add(layers.Dense(16,activation='relu'))
model_SGD.add(layers.Dense(16,activation='relu'))
model_SGD.add(layers.Dense(3,activation='softmax'))
sgd_opt=keras.optimizers.SGD(learning_rate=0.01,momentum=0,nesterov=False)
model_SGD.compile(optimizer=sgd_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#150
history_SGD=model_SGD.fit(x_train,y_train,epochs=150,batch_size=4,validation_data=(x_val,y_val))
history_dict_SGD=history_SGD.history
model_SGD.save("sgd_optimizer.hdf5")
loss_accuracy_plot(history_dict_SGD,name='SGD Optimizer')

model_new_SGD=load_model("sgd_optimizer.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_SGD=model_new_SGD.evaluate(x_test,y_test)
print('Loss: {}'.format(result_SGD[0]))
print('Accuracy: {}'.format(result_SGD[1]))

#RMS Prop 
model_RMS_PROP=models.Sequential()
model_RMS_PROP.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_RMS_PROP.add(layers.Dense(64,activation='relu'))
model_RMS_PROP.add(layers.Dense(3,activation='softmax'))
rmsprop_opt=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model_RMS_PROP.compile(optimizer=rmsprop_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#5
history_RMS_PROP=model_RMS_PROP.fit(x_train,y_train,epochs=20,batch_size=8,validation_data=(x_val,y_val))
history_dict_RMS_PROP=history_RMS_PROP.history
model_RMS_PROP.save("rmsprop.hdf5")
loss_accuracy_plot(history_dict_RMS_PROP,name='RMS Prop')

model_new_RMS_PROP=load_model("rmsprop.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_RMS_PROP=model_new_RMS_PROP.evaluate(x_test,y_test)
print('Loss: {}'.format(result_RMS_PROP[0]))
print('Accuracy: {}'.format(result_RMS_PROP[1]))


#AdaGrad
model_AdaGrad=models.Sequential()
model_AdaGrad.add(layers.Dense(32,activation='relu',input_shape=(4,)))
model_AdaGrad.add(layers.Dense(16,activation='relu'))
model_AdaGrad.add(layers.Dense(16,activation='relu'))
model_AdaGrad.add(layers.Dense(3,activation='softmax'))
adagrad_opt=keras.optimizers.Adagrad(learning_rate=0.01,epsilon=0.5,decay=0.0) 
model_AdaGrad.compile(optimizer=adagrad_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#47
history_AdaGrad=model_AdaGrad.fit(x_train,y_train,epochs=47,batch_size=1,validation_data=(x_val,y_val))
history_dict_AdaGrad=history_AdaGrad.history
model_AdaGrad.save("adagrad.hdf5")
loss_accuracy_plot(history_dict_AdaGrad,name='AdaGrad')

#Loading saved Models
model_new_AdaGrad=load_model("adagrad.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_AdaGrad=model_new_AdaGrad.evaluate(x_test,y_test)
print('Loss: {}'.format(result_AdaGrad[0]))
print('Accuracy: {}'.format(result_AdaGrad[1]))




#AdaDelta
model_AdaDelta=models.Sequential()
model_AdaDelta.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_AdaDelta.add(layers.Dense(16,activation='relu'))
model_AdaDelta.add(layers.Dense(64,activation='relu'))
model_AdaDelta.add(layers.Dense(3,activation='softmax'))
adadelta_opt=keras.optimizers.Adadelta(learning_rate=0.01,rho=0.95,epsilon=0.5,decay=0.0) 
model_AdaDelta.compile(optimizer=adadelta_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#300
history_AdaDelta=model_AdaDelta.fit(x_train,y_train,epochs=100,batch_size=4,validation_data=(x_val,y_val))
history_dict_AdaDelta=history_AdaDelta.history
model_AdaDelta.save("adadelta.hdf5")
#print("Model Saved") 
loss_accuracy_plot(history_dict_AdaDelta,name='AdaDelta')

#Loading saved Models
model_new=load_model("adadelta.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_AdaDelta=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result_AdaDelta[0]))
print('Accuracy: {}'.format(result_AdaDelta[1]))





#Adam
model_Adam=models.Sequential()
model_Adam.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_Adam.add(layers.Dense(16,activation='relu'))
model_Adam.add(layers.Dense(64,activation='relu'))
model_Adam.add(layers.Dense(64,activation='relu'))
model_Adam.add(layers.Dense(3,activation='softmax'))
adam_opt=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.5,decay=0.0,amsgrad=False)
model_Adam.compile(optimizer=adam_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#200
history_Adam=model_Adam.fit(x_train,y_train,epochs=40,batch_size=4,validation_data=(x_val,y_val))
history_dict_Adam=history_Adam.history
model_Adam.save("adam.hdf5")
loss_accuracy_plot(history_dict_Adam,name='Adam')

model_new=load_model("adam.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_Adam=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result_Adam[0]))
print('Accuracy: {}'.format(result_Adam[1]))



#AdaMax
model_AdaMax=models.Sequential()
model_AdaMax.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_AdaMax.add(layers.Dense(16,activation='relu'))
model_AdaMax.add(layers.Dense(64,activation='relu'))
model_AdaMax.add(layers.Dense(64,activation='relu'))
model_AdaMax.add(layers.Dense(3,activation='softmax'))
adamax_opt=keras.optimizers.Adamax(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.1 ,decay=0.0)
model_AdaMax.compile(optimizer=adamax_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#60
history_AdaMax=model_AdaMax.fit(x_train,y_train,epochs=15,batch_size=8,validation_data=(x_val,y_val))
history_dict_AdaMax=history_AdaMax.history
model_AdaMax.save("adamax.hdf5")
loss_accuracy_plot(history_dict_AdaMax,name='AdaMax Optimizer')


model_new_AdaMax=load_model("adamax.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_AdaMax=model_new_AdaMax.evaluate(x_test,y_test)
print('Loss: {}'.format(result_AdaMax[0]))
print('Accuracy: {}'.format(result_AdaMax[1]))




#Nadam Optimizer
model_Nadam=models.Sequential()
model_Nadam.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_Nadam.add(layers.Dense(16,activation='relu'))
model_Nadam.add(layers.Dense(64,activation='relu'))
model_Nadam.add(layers.Dense(64,activation='relu'))
model_Nadam.add(layers.Dense(3,activation='softmax'))
nadam_opt=keras.optimizers.Nadam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.1,schedule_decay=0.004)
model_Nadam.compile(optimizer=nadam_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
#80
history_Nadam=model_Nadam.fit(x_train,y_train,epochs=9,batch_size=8,validation_data=(x_val,y_val))
history_dict_Nadam=history_Nadam.history
model_Nadam.save("nadamax.hdf5")
loss_accuracy_plot(history_dict_Nadam,name='NAdaMax Optimizer')

model_new_Nadam=load_model("nadamax.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_Nadam=model_new_Nadam.evaluate(x_test,y_test)
print('Loss: {}'.format(result_Nadam[0]))
print('Accuracy: {}'.format(result_Nadam[1]))


#Evaluating Diffrent Reguralization Measures

#Weight Decay on Adam 
model_weight_decay=models.Sequential()
model_weight_decay.add(layers.Dense(16,activation='relu',input_shape=(4,),kernel_regularizer=regularizers.l2(0.05)))
model_weight_decay.add(layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
model_weight_decay.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
model_weight_decay.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
model_weight_decay.add(layers.Dense(3,activation='softmax'))
adam_opt=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.5,decay=0.0,amsgrad=False)
model_weight_decay.compile(optimizer=adam_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history_weight_decay=model_weight_decay.fit(x_train,y_train,epochs=500,batch_size=4,validation_data=(x_val,y_val))
history_dict_weight_decay=history_weight_decay.history
model_weight_decay.save("weight_decay.hdf5")
loss_accuracy_plot(history_dict_weight_decay,name='Weight-Decay')

model_new_weight_decay=load_model("weight_decay.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_weight_decay=model_new_weight_decay.evaluate(x_test,y_test)
print('Loss: {}'.format(result_weight_decay[0]))
print('Accuracy: {}'.format(result_weight_decay[1]))




#Dropout with RMSProp regularizer :
#-------------------------------------------------------------------------------

model_dropout=models.Sequential()
model_dropout.add(layers.Dense(16,activation='relu',input_shape=(4,)))
model_dropout.add(layers.Dropout(0.5))
model_dropout.add(layers.Dense(16,activation='relu'))
model_dropout.add(layers.Dropout(0.5))
model_dropout.add(layers.Dense(3,activation='softmax'))
rmsprop_opt=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model_dropout.compile(optimizer=rmsprop_opt,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history_dropout=model_dropout.fit(x_train,y_train,epochs=130,batch_size=4,validation_data=(x_val,y_val))
history_dict_dropout=history_dropout.history
model_dropout.save("dropout.hdf5")
loss_accuracy_plot(history_dict_dropout,name='Drop-Out')

model_new_dropout=load_model("dropout.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result_dropout=model_new_dropout.evaluate(x_test,y_test)
print('Loss: {}'.format(result_dropout[0]))
print('Accuracy: {}'.format(result_dropout[1]))





#Batch Normalization with SGD Optimizer Classifer

model = models.Sequential()
#Layer1
model.add(layers.Dense(16, input_dim=4, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Layer2
model.add(layers.Dense(16, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Output Layer
model.add(layers.Dense(3, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

sgd=keras.optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True)
model.compile(optimizer=sgd,loss=losses.categorical_crossentropy,metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=100,batch_size=16,validation_data=(x_val,y_val))
history_dict=history.history
model.save("batch_norm_sgd.hdf5")
loss_accuracy_plot(history_dict,name='Batch Norm')

model_new=load_model("batch_norm_sgd.hdf5")

#Evaluating Model on test Data.
print("Results on Test Data")
result=model_new.evaluate(x_test,y_test)
print('Loss: {}'.format(result[0]))
print('Accuracy: {}'.format(result[1]))



#Ensemble Classifier using two models : Adam Classifier and RMSProp Classifier
#Loading Models
model_adam=load_model("adam.hdf5")
model_rmsprop=load_model("rmsprop.hdf5")

#Getting Predicated Values from Two models
predicted_adam=model_adam.predict(x_test)
predicted_rmsprop=model_rmsprop.predict(x_test)

#Average Predicted Values
avg_predicted_y=(predicted_adam+predicted_rmsprop)/2.0

#Predicating Labels
pred_label=[]
for i in avg_predicted_y:
        max_value = np.max(i)
        pred_label.append(list(np.where(i==max_value,1,0)))  
    
#Getting Accuracy 
count = 0
for i,j in zip(y_test,pred_label):
        if ((i[0] == j[0]) & (i[1] == j[1]) & (i[2] == j[2])):
            count +=1

acc = ((count/x_test.shape[0]))
print("Accuracy of an Ensemble Classifier {}".format(acc))