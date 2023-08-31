#Importing the required libs for the exercise

from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN


class TimeGANAux():
    def __init__(self, train_data, label, seq_len=11, n_seq=626, hidden_dim=24, gamma=1, noise_dim=32, dim=128, batch_size=30, learning_rate=5e-4):
        self.train_data = np.copy(train_data)
        self.label = label
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.gan_args = ModelParameters(batch_size=batch_size,
                                        lr=learning_rate,
                                        noise_dim=noise_dim,
                                        layers_dim=dim)
        self.synth = None
    
    def train(self, train_steps=300):
        file_name = f'data/synthesizer_eeg__label_{self.label}_batch_size_{self.batch_size}_train_steps_{train_steps}.pkl'
        if path.exists(file_name):
            self.synth = TimeGAN.load(file_name)
        else:
            self.synth = TimeGAN(model_parameters=self.gan_args, hidden_dim=self.hidden_dim, seq_len=self.seq_len, n_seq=self.n_seq, gamma=self.gamma)
            self.synth.train(self.train_data, train_steps=train_steps)
            self.synth.save(file_name)
    
    def generate(self, num_of_batches=1):
        synth_data = self.synth.sample(num_of_batches*self.batch_size - 1)
        return synth_data
