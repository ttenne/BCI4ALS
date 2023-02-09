import numpy as np
import tensorflow as tf
# from tensorflow.keras import datasets, layers, models

# def denoise(MIData):
#     print(MIData.shape)
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Conv1D(8, 3, activation='relu', input_shape=(11, 626)))
#     model.add(tf.keras.layers.Conv1D(16, 3, activation='relu'))
#     model.add(tf.keras.layers.Conv1D(32, 3, activation='relu'))
#     model.add(tf.keras.layers.Conv1DTranspose(32, 3, activation='relu'))
#     model.add(tf.keras.layers.Conv1DTranspose(16, 3, activation='relu'))
#     model.add(tf.keras.layers.Conv1DTranspose(8, 3, activation='relu'))
#     model.summary()
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.losses.MeanSquaredError)
#     MIData = [[[0 for _ in range(626)] for _ in range(11)] for _ in range(120)]
#     MIData = np.array(MIData)
#     print(MIData.shape)
#     history = model.fit(x=MIData, y=MIData, epochs=1)
#     # plt.plot(history.history['accuracy'], label='accuracy')
#     exit()

def denoise(MIData, filters_per_layer=[8,16,32], kernel_sizes=[5,5,3], epochs=3):
    # kernel_size = MIData.shape[1]
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters_per_layer[0], kernel_size=kernel_sizes[0], activation='relu', input_shape=(MIData.shape[1:]))) #input_shape = (batch_size, feature_size, channels)
    for i, filter_num in enumerate(filters_per_layer[1:]):
        model.add(tf.keras.layers.Conv1D(filter_num, kernel_size=kernel_sizes[1:][i], activation='relu'))
    for i, filter_num in enumerate(filters_per_layer[::-1]):
        model.add(tf.keras.layers.Conv1DTranspose(filter_num, kernel_size=kernel_sizes[::-1][i], activation='relu'))
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.losses.MeanSquaredError())
    history = model.fit(x=MIData, y=MIData, epochs=epochs)
    # plt.plot(history.history['accuracy'], label='accuracy')

def check_reshape(orig, reshaped):
    dim1 = len(orig) #trials
    dim2 = len(orig[0]) #electrodes
    dim3 = len(orig[0][0]) #samples
    for i, sample in enumerate(reshaped):
        for electrode in range(len(sample)):
            if sample[electrode] != orig[i//dim3][electrode][i%dim3]:
                print('reshape failed')
                return
    print('reshape succeded')

def reshapeMIData(orig):
    dim1 = len(orig) #trials
    dim2 = len(orig[0]) #electrodes
    dim3 = len(orig[0][0]) #samples
    reshaped = [[[] for _ in range(dim3)] for _ in range(dim1)]
    for i, trial in enumerate(orig):
        for electrode in trial:
            for j, sample in enumerate(electrode):
                reshaped[i][j].append(sample)
    return np.array(reshaped)

fake_MIData = np.array([[[np.random.normal() for _ in range(626)] for _ in range(11)] for _ in range(120)])
# fake_MIData = np.array([[[np.random.normal() for _ in range(5)] for _ in range(2)] for _ in range(3)])
# print(fake_MIData.shape)
# print(fake_MIData)
# fake_MIData_reshaped = reshapeMIData(fake_MIData)
# print(fake_MIData_reshaped.shape)
# print(fake_MIData_reshaped)
# check_reshape(fake_MIData, fake_MIData_reshaped)
# print(fake_MIData_reshaped.shape)
filtered_fake_MIData = denoise(fake_MIData)
