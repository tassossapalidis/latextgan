import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# defines ResNet architecture for Generator and Discriminator
class ResNetBlock(keras.Model):
    def __init__(self, block_dim):
        super().__init__()

        # linear layer with ReLU activation function
        self.layer1 = keras.layers.Dense(block_dim, activation = 'relu')
        # linear layer
        self.layer2 = keras.layers.Dense(block_dim)
    
    def call(self, x):
        # ResNet: input is added to layer output
        x = self.layer2(self.layer1((x))) + x
        return x

class Generator(keras.Sequential):
    def __init__(self, block_dim, n_layers):
        super(Generator, self).__init__()

        # simple sequence of ResNet blocks
        for i in range(n_layers):
            self.add(ResNetBlock(block_dim))

class Discriminator(keras.Sequential):
    def __init__(self, n_layers, block_dim):
        super(Discriminator, self).__init__()

        # simple sequence of ResNet blocks
        for i in range(n_layers):
            self.add(ResNetBlock(block_dim))

if __name__ == '__main__':
    # create a generator
    generator = Generator(100, 20)
    
    # produce output using normal noise
    noise = np.random.normal(size = (1, 100))
    z = generator(noise[None, :, :])

    print(z)
