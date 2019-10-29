
# coding: utf-8

# In[2]:


import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler

usps_data = pickle.load(open('D:/UTD OneDrive/OneDrive - The University of Texas at Dallas/UTD/Sem 1/CS6375.001 Machine Learning/Programming Assignments/Programming Assignment 3/usps.pickle', 'rb'))
x_val=usps_data['x']['val']
x_trn=usps_data['x']['trn']
x_tst=usps_data['x']['tst']

y_val=usps_data['y']['val']
y_trn=usps_data['y']['trn']
y_tst=usps_data['y']['tst']

def build_model(n_kernels, kernel_size, stride, n_dense):
    model = tf.keras.models.Sequential()
    model.add(layers.Convolution2D(filters=n_kernels,kernel_size=(kernel_size,kernel_size),activation='relu',input_shape=(16,16,1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(strides=(stride,stride)))
    model.add(layers.Dropout(rate=0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_dense,activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(10,activation='softmax'))
    
    adamOptimizer= tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adamOptimizer,loss='categorical_crossentropy')
    
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    history = model.fit(x_trn,y_trn,epochs=2,batch_size =16, verbose=2, 
                        validation_data=(x_val, y_val),callbacks=[annealer])
    
    tstError=model.evaluate(x_tst,y_tst)
    trnError=model.evaluate(x_trn,y_trn)    
    noOfParams=model.count_params()
    return (tstError,trnError,noOfParams)

"""a. (The effect of Filter Size, 25 points) Keeping all the other parameters fixed to their default values, train
models with n kernels = 1; 2; 4; 8; 16, and evaluate them on the test set. Plot (i) n kernels vs. training
and test error, and (ii) n kernels vs. total number of network parameters. What seems to be a reasonable
choice of n kernels?"""

tstErrorArr=[]
trnErrorArr=[]
noOfParamArr=[]

tstError,trnError,noOfParams= build_model(n_kernels=1,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=2,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=4,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=16,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

print("Test Errors:",tstErrorArr)
print("Training Errors:",trnErrorArr)
print("no of Params:",noOfParamArr)
nKernels=[1,2,4,8,16]
plt.plot(nKernels,trnErrorArr)
plt.xlabel("n_kernels")
plt.ylabel("Training error")
plt.show()

plt.plot(nKernels,tstErrorArr)
plt.xlabel("n_kernels")
plt.ylabel("Testing error")
plt.show()

plt.plot(nKernels,noOfParamArr)
plt.xlabel("n_kernels")
plt.ylabel("No of Parameters")
plt.show()


"""b. (The effect of Kernel Size, 25 points) Keeping all the other parameters fixed to their default values,
train models with kernel size = 1; 2; 3; 4; 5, and evaluate them on the test set. Plot (i) kernel size vs.
training and test error, and (ii) kernel size vs. total number of network parameters. What seems to be a
reasonable choice of kernel size?"""
tstErrorArr=[]
trnErrorArr=[]
noOfParamArr=[]

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=1,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=2,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=4,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=5,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

print("Test Errors:",tstErrorArr)
print("Training Errors:",trnErrorArr)
print("no of Params:",noOfParamArr)
kernelSize=[1,2,3,4,5]
plt.plot(kernelSize,trnErrorArr)
plt.xlabel("kernel_size")
plt.ylabel("Training error")
plt.show()

plt.plot(kernelSize,tstErrorArr)
plt.xlabel("kernal_size")
plt.ylabel("Testing error")
plt.show()

plt.plot(kernelSize,noOfParamArr)
plt.xlabel("kernal_size")
plt.ylabel("No of Parameters")
plt.show()

"""c. (The effect of Stride, 25 points) Keeping all the other parameters fixed to their default values, train
models with stride = 1; 2; 3; 4. Plot (i) stride vs. training and test error, and (ii) stride vs. total number
of network parameters, and evaluate them on the test set. What seems to be a reasonable choice of stride?"""
tstErrorArr=[]
trnErrorArr=[]
noOfParamArr=[]

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=1,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=3,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=4,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

print("Test Errors:",tstErrorArr)
print("Training Errors:",trnErrorArr)
print("no of Params:",noOfParamArr)
strides=[1,2,3,4]
plt.plot(strides,trnErrorArr)
plt.xlabel("strides")
plt.ylabel("Training error")
plt.show()

plt.plot(strides,tstErrorArr)
plt.xlabel("strides")
plt.ylabel("Testing error")
plt.show()

plt.plot(strides,noOfParamArr)
plt.xlabel("strides")
plt.ylabel("No of Parameters")
plt.show()


"""d. (The effect of Dense Layer Size, 25 points) Keeping all the other parameters fixed to their default values,
train models with n dense = 16; 32; 64; 128. Plot (i) n dense vs. training and test error, and (ii) n dense
vs. total number of network parameters, and evaluate them on the test set. What seems to be a reasonable
choice of n dense?"""
tstErrorArr=[]
trnErrorArr=[]
noOfParamArr=[]

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=2,n_dense =16)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=2,n_dense =32)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=2,n_dense =64)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

tstError,trnError,noOfParams= build_model(n_kernels=8,kernel_size=3,stride=2,n_dense =128)
print(tstError,trnError,noOfParams)
tstErrorArr.append(tstError)
trnErrorArr.append(trnError)
noOfParamArr.append(noOfParams)

print("Test Errors:",tstErrorArr)
print("Training Errors:",trnErrorArr)
print("no of Params:",noOfParamArr)
nDense=[16,32,64,128]
plt.plot(nDense,trnErrorArr)
plt.xlabel("nDense")
plt.ylabel("Training error")
plt.show()


plt.plot(nDense,tstErrorArr)
plt.xlabel("nDense")
plt.ylabel("Testing error")
plt.show()

plt.plot(nDense,noOfParamArr)
plt.xlabel("nDense")
plt.ylabel("No of Parameters")
plt.show()


