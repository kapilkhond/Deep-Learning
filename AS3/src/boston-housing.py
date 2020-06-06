# -*- coding: utf-8 -*-
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
from sklearn import preprocessing
from keras.datasets import boston_housing
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_label), (test_data, test_targets) = boston_housing.load_data()

#normalise data 
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

def model_building(optim,loss):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optim,loss=loss,metrics=['mean_absolute_error'])
    return model

def k_fold_cross_val(optim,loss,epoch,batch_size):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold no. {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_label[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_label = np.concatenate(
                                [train_label[:i * num_val_samples],
                                train_label[(i+1)*num_val_samples:]],
                                axis=0)        
        model = model_building(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_label,
                  epochs=num_epochs,
                  batch_size=batch_size,validation_data=(val_data,val_targets)) #
        train_score.append(history.history['mean_absolute_error'])
        val_score.append(history.history['val_mean_absolute_error'])
    
    
    
    #Compute average mean absolute error
    avg_mean_absolute_error_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mean_absolute_error_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 
    return avg_mean_absolute_error_history,avg_val_mean_absolute_error_history



#Mean Square Error
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
loss=losses.mean_squared_error
epoch=80
batch_size=1

avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
#plt.ylim(list(range(2,4)))
plt.show()



#Final Model Building 
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("mean_squared_error.hdf5")

#Loading Saved model
model_new=load_model("mean_squared_error.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))




#Mean Absolute Error
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
loss=losses.mean_absolute_error
epoch=120
batch_size=1


avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.title('Mean Absolute Error')
plt.show()



#Bulding Final Model
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

model_final.save("mean_absolute_error.hdf5")

model_new=load_model("mean_absolute_error.hdf5")

#Mean Square Error Loss Function Evaluation 
test_loss,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("Loss Value: {}".format(test_loss))
print("Mean Absolute error: {}".format(test_mean_absolute_error))




#Mean Absolute Percentage Error
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
loss=losses.mean_absolute_percentage_error
epoch=100
batch_size=4


avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.title('Mean Absolute Percentage Error')
plt.show()


#Bulding Final Model
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

model_final.save("mean_absolute_percentage_error.hdf5")

model_new=load_model("mean_absolute_percentage_error.hdf5")

#Evaluating Mean Square Error Loss Function
test_loss,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("Loss Value: {}".format(test_loss))
print("Mean Absolute error: {}".format(test_mean_absolute_error))


#Mean Square Logarithmic Error
#Defining Optimizer and Loss Function 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
loss=losses.mean_squared_logarithmic_error
epoch=200
batch_size=4

avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae Validation Error")
plt.title('Mean Square Logarithmic Error')
plt.show()


#Bulding Final Model
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
print("Saving the Model...")
model_final.save("mean_squared_logarithmic_error.hdf5")

#Loading Saved model
print("Loading the Model...")
model_new=load_model("mean_squared_logarithmic_error.hdf5")

#Evaluating Mean Square Error Loss Function
test_loss,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("Loss Value: {}".format(test_loss))
print("Mean Absolute error: {}".format(test_mean_absolute_error))




#Evaluating Diffrent Optimizers with MSE loss
# 1. SGD Optimizer

sgd=keras.optimizers.SGD(learning_rate=0.001,momentum=0.0,nesterov=False)
loss=losses.mean_squared_error
epoch=100
batch_size=4

avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(sgd,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.title('SGD Optimizer')
plt.show()



#Bulding Final Model
model_final=model_building(sgd,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("sgd.hdf5")

#Loading Saved model
model_new=load_model("sgd.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)
print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))




#RMS Prop 
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

loss=losses.mean_squared_error
epoch=50
batch_size=1


avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.title('RMSProp Optimizer')
plt.show()

#Bulding Final Model
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("rmsprop.hdf5")

#Loading Saved model
model_new=load_model("rmsprop.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)
print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))



#AdaGrad 
adagrad=keras.optimizers.Adagrad(learning_rate=0.1,epsilon=0.5,decay=0.0) 
loss=losses.mean_squared_error
epoch=50
batch_size=4


avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(adagrad,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.title('AdaGrad Optimizer')
plt.show()



#Bulding Final Model
model_final=model_building(adagrad,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("adagrad.hdf5")

#Loading Saved model
model_new=load_model("adagrad.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))



#adadelta_opt_model
adadelta_opt_model=keras.optimizers.Adadelta(learning_rate=0.01,rho=0.95,epsilon=0.5,decay=0.0)
loss=losses.mean_squared_error
epoch=150
batch_size=1


avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(adadelta_opt_model,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,4])
plt.title('Adadelta Optimizer')
plt.show()


#Bulding Final Model
model_final=model_building(adadelta_opt_model,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("adadelta_opt_model.hdf5")

#Loading Saved model
model_new=load_model("adadelta_opt_model.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))

#adam_opt_model
adam_opt_model=keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.5,decay=0.0,amsgrad=False)

loss=losses.mean_squared_error
epoch=80
batch_size=4


avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(adam_opt_model,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,4])
plt.title('Adam Optimizer')
plt.show()



#Bulding Final Model
model_final=model_building(adam_opt_model,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("adam_opt_model.hdf5")

#Loading Saved model
model_new=load_model("adam_opt_model.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))



#adam_opt_modelax
adam_opt_modelax=keras.optimizers.adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.1 ,decay=0.0)
loss=losses.mean_squared_error
epoch=100
batch_size=4

avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(adam_opt_modelax,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,3.5])
plt.title('Adamax Optimizer')
plt.show()


#Bulding Final Model
model_final=model_building(adam_opt_modelax,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("adam_opt_modelax.hdf5")

#Loading Saved model
model_new=load_model("adam_opt_modelax.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))





#Nadam_opt_model

nadam_opt_model=keras.optimizers.Nadam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=0.1,schedule_decay=0.004)
loss=losses.mean_squared_error
epoch=120
batch_size=4


avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val(nadam_opt_model,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,3])
plt.title('Nadam Optimizer')
plt.show()


#Bulding Final Model
model_final=model_building(nadam_opt_model,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("nadam_opt_model.hdf5")

#Loading Saved model
model_new=load_model("nadam_opt_model.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))


#Weight Decay on RMSProp
def model_building_weight_decay(optim,loss):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],),kernel_regularizer=regularizers.l2(0.05)))
    model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.05)))
    model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.05)))#Temp
    model.add(layers.Dense(1))

    model.compile(optimizer=optim,loss=loss,metrics=['mean_absolute_error'])
              
    return model

def k_fold_cross_validation_weight_decay(optim,loss,epoch,batch_size):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_label[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_label = np.concatenate(
                                [train_label[:i * num_val_samples],
                                train_label[(i+1)*num_val_samples:]],
                                axis=0)
        
        
        #rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
        #loss=losses.mean_squared_error
        
        model = model_building_weight_decay(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_label,
                  epochs=num_epochs,
                  batch_size=batch_size,validation_data=(val_data,val_targets),verbose=0) #
        train_score.append(history.history['mean_absolute_error'])
        val_score.append(history.history['val_mean_absolute_error'])
    
    
    
    #Compute AVg mean_absolute_error
    avg_mean_absolute_error_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mean_absolute_error_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 

    return avg_mean_absolute_error_history,avg_val_mean_absolute_error_history


rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
loss=losses.mean_squared_error
epoch=500
batch_size=4
avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_validation_weight_decay(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MEA Validation Error")
plt.ylim([2,4])
plt.title('Weight Decay Regularization')
plt.show()



#Bulding Final Model
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("rmsprop_wt_decay.hdf5")

#Loading Saved model
model_new=load_model("rmsprop_wt_decay.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))



#Drop Out Regularization
def model_building_dropout(optim,loss):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))#Temp
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))

    model.compile(optimizer=optim,loss=loss,metrics=['mean_absolute_error'])
              
    return model

def k_fold_cross_val_dropout(optim,loss,epoch,batch_size):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_label[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_label = np.concatenate(
                                [train_label[:i * num_val_samples],
                                train_label[(i+1)*num_val_samples:]],
                                axis=0)
        
        
        #rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
        #loss=losses.mean_squared_error
        
        model = model_building_dropout(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_label,
                  epochs=num_epochs,
                  batch_size=batch_size,validation_data=(val_data,val_targets),verbose=0) #
        train_score.append(history.history['mean_absolute_error'])
        val_score.append(history.history['val_mean_absolute_error'])
    
    
    
    #Compute AVg mean_absolute_error
    avg_mean_absolute_error_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mean_absolute_error_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 

    return avg_mean_absolute_error_history,avg_val_mean_absolute_error_history




rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
loss=losses.mean_squared_error
epoch=500
batch_size=4

avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val_dropout(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
import matplotlib.pyplot as plt
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MAE Validation Error")
plt.ylim([2,4])
plt.title('DropOut Regularization')
plt.show()

#Bulding Final Model
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("rmsprop_dropout.hdf5")
#Loading Saved model
model_new=load_model("rmsprop_dropout.hdf5")
#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)


print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))

#Batch Normalization:
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

def model_building_batch_norm(optim,loss):    
    model = models.Sequential()
    
    #Layer - 1 
    model.add(layers.Dense(64, input_shape=(train_data.shape[1],), init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Layer - 2
    model.add(layers.Dense(64, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
            
    # Layer - 3
    model.add(layers.Dense(64, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Output Layer
    model.add(layers.Dense(1, init='uniform'))
    
    model.compile(optimizer=optim,loss=loss,metrics=['mean_absolute_error'])
              
    return model

def k_fold_cross_val_batch_norm(optim,loss,epoch,batch_size):
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = epoch
    
    train_score=[]
    val_score=[]
    for i in range(k):
        print(f'Processing fold # {i}')
        val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
        val_targets = train_label[i * num_val_samples: (i+1) * num_val_samples]
        
        partial_train_data = np.concatenate(
                                [train_data[:i * num_val_samples],
                                train_data[(i+1) * num_val_samples:]],
                                axis=0)
        partial_train_label = np.concatenate(
                                [train_label[:i * num_val_samples],
                                train_label[(i+1)*num_val_samples:]],
                                axis=0)
        
        
        #rmsprop=keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9)
        #loss=losses.mean_squared_error
        
        model = model_building_batch_norm(optim,loss)
        history=model.fit(partial_train_data,
                  partial_train_label,
                  epochs=num_epochs,
                  batch_size=batch_size,validation_data=(val_data,val_targets),verbose=0) #
        train_score.append(history.history['mean_absolute_error'])
        val_score.append(history.history['val_mean_absolute_error'])
    
    
    
    #Compute AVg mean_absolute_error
    avg_mean_absolute_error_history=[np.mean([x[i] for x in train_score]) for i in range(num_epochs)]    
    avg_val_mean_absolute_error_history=[np.mean([y[j] for y in val_score]) for j in range(num_epochs)] 

    return avg_mean_absolute_error_history,avg_val_mean_absolute_error_history




rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
loss=losses.mean_squared_error
epoch=50
batch_size=4
avg_mean_absolute_error_history,avg_val_mean_absolute_error_history=k_fold_cross_val_batch_norm(rmsprop,loss,epoch,batch_size)

#Ploting the Data for Hyperparameter Tuning
plt.plot(range(1,len(avg_val_mean_absolute_error_history)+1),avg_val_mean_absolute_error_history)
plt.xlabel("Epochs")
plt.ylabel("Avergae MAE Validation Error")
plt.ylim([2,7])
plt.title('Batch Normalization Regularization')
plt.show()


#Bulding Final Model
model_final=model_building(rmsprop,loss)
model_final.fit(train_data,train_label,epochs=epoch,batch_size=batch_size)

#Saving The Final Model
model_final.save("rmsprop_bnorm.hdf5")

#Loading Saved model
model_new=load_model("rmsprop_bnorm.hdf5")

#Evaluating Mean Square Error Loss Function
test_mean_squared_error,test_mean_absolute_error=model_new.evaluate(test_data, test_targets)

print("After evaluating on test data:")
print("MSE Loss Value: {}".format(test_mean_squared_error))
print("Mean Absolute error: {}".format(test_mean_absolute_error))

#Ensemble Classifier using two models : adam_opt_model Classifier and RMSProp Classifier
model_new1=load_model("adam_opt_model.hdf5")
model_new2=load_model("rmsprop.hdf5")

#Getting Predicated Values from Two models
predicted_y1=model_new1.predict(test_data)
predicted_y2=model_new2.predict(test_data)

#Average Predicted Values
avg_predicted_y=(predicted_y1+predicted_y2)/2.0

tt=test_targets.reshape((102, 1))
mean_absolute_error = np.sum(np.absolute(tt-avg_predicted_y))/test_targets.shape[0]

print("Mean Absolute Error {}".format(mean_absolute_error))
