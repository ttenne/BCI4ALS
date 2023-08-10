import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output
    
class Generator(nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, data_size),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class GAN():
    def __init__(self, train_data, batch_size=32, lr=0.001):
        train_data_temp = torch.Tensor(train_data)
        self.original_shape = train_data.shape[1:]
        train_data_temp = train_data_temp.reshape((train_data_temp.shape[0], -1))
        self.data_size = train_data_temp.shape[1]
        train_labels = torch.zeros(train_data_temp.shape[0])
        train_set = [
            (train_data_temp[i], train_labels[i]) for i in range(train_data_temp.shape[0])
        ]
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
        self.discriminator = Discriminator(self.data_size)
        self.generator = Generator(self.data_size)

        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=lr)

    def train(self, num_epochs=300, loss_function=nn.BCELoss()):
        for epoch in range(num_epochs):
            for n, (real_samples, _) in enumerate(self.train_loader):
                # Data for training the discriminator
                real_samples_labels = torch.ones((self.batch_size, 1))
                latent_space_samples = torch.randn((self.batch_size, self.data_size))
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((self.batch_size, 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training the discriminator
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((self.batch_size, self.data_size))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                self.optimizer_generator.step()

                # Show loss
                if epoch % 10 == 0 and n == self.batch_size - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")

    def generate(self, num_of_trials=30):
        latent_space_samples = torch.randn(num_of_trials, self.data_size)
        generated_samples = self.generator(latent_space_samples)

        generated_samples = generated_samples.detach()
        print(f'generated_samples.shape = {generated_samples.shape}')
        generated_samples = generated_samples.reshape((num_of_trials, self.original_shape[0], self.original_shape[1]))
        print(f'generated_samples.shape = {generated_samples.shape}')
        return generated_samples
