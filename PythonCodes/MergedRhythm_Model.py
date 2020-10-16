
import numpy
import keras
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Embedding, Bidirectional, Flatten,BatchNormalization
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras import regularizers
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D,Lambda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import h5py
from keras.layers import LeakyReLU
from sklearn.preprocessing import RobustScaler,Normalizer
from keras import initializers
from sklearn.decomposition import PCA,FastICA


hf = h5py.File('mdataa1.h5', 'r') # Lead-II channel data
X = hf.get('mdataaset_1') # Lead-II
X=np.array(X)
hf.close()
hf = h5py.File('label11.h5', 'r') # Labels
Y = hf.get('label11')
Y=np.array(Y)
hf.close()

hf = h5py.File('Features.h5', 'r') # Clinical Features
feat = hf.get('Features') 
feat=np.array(feat)
hf.close()   
    
    
X=np.reshape(X,(10588,5000)) # Shape
encoder = LabelEncoder()
encoder.fit(Y);
encoded_Y = encoder.transform(Y)
Y = np_utils.to_categorical(encoded_Y,num_classes=4)




X=X.reshape(X.shape[0],X.shape[1],1);

seed =23
numpy.random.seed(seed)
indices = np.arange(X.shape[0])
x_train, X1, y_train, Y1,idx1,idx2 = train_test_split(X, Y,indices, test_size=0.2, random_state=seed)
x_val, x_test, y_val, y_test,idx3,idx4 = train_test_split(X1, Y1,idx2, test_size=0.5, random_state=seed)


adam=keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-3, amsgrad=False)


model = Sequential()
model.add(Conv1D(64,21, strides=11, input_shape = (X.shape[1],1)))
model.add(MaxPooling1D(strides=2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Conv1D(64,7, strides=1, activation = 'relu'))
model.add(MaxPooling1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(128,5, strides=1, activation = 'relu'))
model.add(MaxPooling1D(strides=2))
model.add(Conv1D(256,13, strides=1, activation = 'relu'))
model.add(Conv1D(512,7, strides=1, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Conv1D(256,9, strides=1, activation = 'relu'))
model.add(MaxPooling1D(strides=2))
model.add(LSTM(128,return_sequences=True,dropout=0.0, recurrent_dropout=0.0))
model.summary()
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def katman(lnum,X):
# with a Sequential model
        get_3rd_layer_output = K.function([model.layers[0].input],
                                          [model.layers[lnum].output])
        
        layer_output = get_3rd_layer_output([X])[0]
        pca = PCA(n_components = 1)
        
        X_pca = pca.fit_transform(layer_output[:,:,0])
        for i in range(1,layer_output.shape[2]): #Filter size
            X_pca2 = pca.fit_transform(layer_output[:,:,i])
   
            X_pca=np.column_stack((X_pca,X_pca2));

        X_deepfeat=X_pca;
        seed = 23
        np.random.seed(seed)
        indices = np.arange(X.shape[0])
        
        hf = h5py.File('label11.h5', 'r') #Labels
        Y = hf.get('label11') # 
        Y=np.array(Y)
        hf.close()
        
        Y = np.array(LabelEncoder().fit_transform(Y));
        
        x_train,x_test,y_train,y_test,feat_train,feat_test=train_test_split(X_deepfeat,Y,feat,test_size=0.2, random_state=seed)
        
        
        x_train=np.column_stack((x_train,feat_train)); # Change this section for each experiment this is fused experiment 
        x_test=np.column_stack((x_test,feat_test)); # Change this section for each experiment  this is fused experiment
        
        X=np.concatenate((x_train, x_test));
        Y=np.concatenate((y_train,y_test));

        import Classes
        
        Classes.siniflar(X,Y) # Call shallow classifiers.


# DNN model training for each epoch
for i in range(1,25):
    history=model.fit(x_train, y_train,
                      batch_size=128,
                      epochs=1,
                      validation_data=(X1, Y1),shuffle=True) #callbacks=checkpoints
    score, acc = model.evaluate(X1, Y1,batch_size=128)

    convlayer=13;
    katman(convlayer,X) # Call function

    
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])



