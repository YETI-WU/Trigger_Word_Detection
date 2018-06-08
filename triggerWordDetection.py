# triggerWordDetection.py
"""
Program to train the model for Trigger Word Detection
deeplearning.ai course project
@author: YEN TIEN WU
"""

import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt


Tx = 5511 # The number of time steps input to the model from the spectrogram
Ty = 1375 # The number of time steps in the output of our model
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    nfft = 200 # Length of each window segment 
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim # Number of array dimension
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data,      nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx



#from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Conv1D, BatchNormalization, Activation, Dropout, GRU, Dense, Input, TimeDistributed
from keras.optimizers import Adam


def model(input_shape):
    """
    Function creating the model's graph in Keras. Uni-Directional RNN
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)  # CONV1D inputs 5511 step spectrogram, output 1375 step, (n-f)/s+1
    X = BatchNormalization()(X)              # Batch normalization
    X = Activation('relu')(X)                # ReLu activation
    X = Dropout(0.8)(X)                      # dropout (use 0.8)

    # Step 2: First GRU Layer 
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                      # dropout (use 0.8)
    X = BatchNormalization()(X)              # Batch normalization
    
    # Step 3: Second GRU Layer 
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                      # dropout (use 0.8)
    X = BatchNormalization()(X)              # Batch normalization
    X = Dropout(0.8)(X)                      # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer 
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    
    return model  




model = model(input_shape = (Tx, n_freq))
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


train_history = model.fit(X, Y, batch_size = 5, epochs=50)
#print(train_history.history.keys())

# Plot history of loss
plt.plot(train_history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_history_loss'], loc='upper right')
plt.show()

# plot history of acc
plt.plot(train_history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_history_acc'], loc='lower right')
plt.show()


# Save model for future training
model.save('./models/model_tr_5000epo.h5') 

# Load model from previous trained model, and train again
#model = load_model('./models/tr_model.h5')


loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


