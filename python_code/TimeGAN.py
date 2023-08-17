#Importing the required libs for the exercise

from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.synthesizers.timeseries import TimeGAN


class TimeGANAux():
    def __init__(self, train_data, seq_len=11, n_seq=626, hidden_dim=24, gamma=1, noise_dim=32, dim=128, batch_size=30, learning_rate=5e-4):
        self.seq_len = seq_len
        self.n_seq = n_seq
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.noise_dim = noise_dim
        self.gan_args = ModelParameters(batch_size=batch_size,
                                        lr=learning_rate,
                                        noise_dim=noise_dim,
                                        layers_dim=dim)
        self.train_data = np.copy(train_data)
        self.synth = None
    
    def train(self, train_steps=3000):
        if path.exists('data/synthesizer_eeg.pkl'):
            self.synth = TimeGAN.load('data/synthesizer_eeg.pkl')
        else:
            self.synth = TimeGAN(model_parameters=self.gan_args, hidden_dim=self.hidden_dim, seq_len=self.seq_len, n_seq=self.n_seq, gamma=self.gamma)
            self.synth.train(self.train_data, train_steps=train_steps)
            self.synth.save('data/synthesizer_eeg.pkl')
    
    def generate(self, num_of_trials=30):
        synth_data = self.synth.sample(num_of_trials)
        return synth_data
