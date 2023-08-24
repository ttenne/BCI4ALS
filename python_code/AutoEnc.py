import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path

class AutoEncoder():

    def __init__(self, MIData, filters_per_layer=[32,16,8], kernel_sizes=[5,5,5], epochs=250, load_weights=True):
        self.model = self.getModel(MIData, filters_per_layer, kernel_sizes, epochs, load_weights, show_train_error=False)
        
    def reshapeMIData(self, orig):
        dim1 = len(orig) #trials
        dim2 = len(orig[0]) #electrodes
        dim3 = len(orig[0][0]) #samples
        reshaped = [[[] for _ in range(dim3)] for _ in range(dim1)]
        for i, trial in enumerate(orig):
            for electrode in trial:
                for j, sample in enumerate(electrode):
                    reshaped[i][j].append(sample)
        return np.array(reshaped)
    
    def getModel(self, MIData, filters_per_layer, kernel_sizes, epochs, load_weights, show_train_error):
        reshaped_MIData = self.reshapeMIData(MIData)
        train_val_split_percent = 1 #0.8
        train_MIData = reshaped_MIData[:int(train_val_split_percent*len(reshaped_MIData))]
        val_MIData = reshaped_MIData[len(train_MIData):]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(filters_per_layer[0], kernel_size=kernel_sizes[0], activation='relu', input_shape=(train_MIData.shape[1:]))) #input_shape = (batch_size, feature_size, channels)
        for i, filter_num in enumerate(filters_per_layer[1:]):
            model.add(tf.keras.layers.Conv1D(filter_num, kernel_size=kernel_sizes[1:][i], activation='relu'))
        for i, filter_num in enumerate(filters_per_layer[::-1]):
            model.add(tf.keras.layers.Conv1DTranspose(filter_num, kernel_size=kernel_sizes[::-1][i], activation='relu'))
        model.add(tf.keras.layers.Conv1D(train_MIData.shape[-1], 1))#, activation='relu'))
        # model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.losses.MeanSquaredError())
        if load_weights and path.exists('data/auto_encoder_weights.h5'):
            model.load_weights('data/auto_encoder_weights.h5')
        else:
            history = model.fit(x=train_MIData, y=train_MIData, epochs=epochs, validation_data=(val_MIData, val_MIData))
            model.save_weights('data/auto_encoder_weights.h5', overwrite=True)
            if show_train_error:
                plt.plot(history.history['loss'], label='train_loss')
                if train_val_split_percent < 1:
                    plt.plot(history.history['val_loss'], label='validation_loss')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.title('Auto Encoder CNN Training')
                plt.legend()
                plt.tight_layout()
                plt.show()
                plt.clf()
        return model

    def predict(self, MIData):
        reshaped_MIData = self.reshapeMIData(MIData)
        return self.reshapeMIData(self.model.predict(reshaped_MIData))
