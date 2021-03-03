import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# defines ResNet architecture for Generator and Discriminator
class ResNetBlock(keras.Model):
    def __init__(self, block_dim):
        super().__init__()

        # linear layer with ReLU activation function
        self.layer1 = keras.layers.Dense(block_dim, activation = 'relu', kernel_initializer=tf.keras.initializers.HeNormal())
        # linear layer
        self.layer2 = keras.layers.Dense(block_dim)
        self.relu = keras.layers.ReLU()

    def call(self, x):
        '''# ResNet: input is added to layer output
        x = self.layer2(self.layer1((x))) + x
        #return self.relu(x)
        return x'''
        output = x
        output = tf.nn.relu(output)
        output = self.layer1(output)
        output = tf.nn.relu(output)
        output = self.layer2(output)
        return x + (0.3*output)
        #return x + output

class ResNetLeakyRelu(keras.Model):
    def __init__(self, block_dim):
        super().__init__()

        # linear layer with ReLU activation function
        self.layer1 = keras.layers.Dense(block_dim, activation = "linear", kernel_initializer=tf.keras.initializers.HeNormal())
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.layer2 = keras.layers.Dense(block_dim, kernel_initializer=tf.keras.initializers.HeNormal())
        self.relu = keras.layers.ReLU()

    def call(self, x):
        x = self.layer2(self.leaky_relu(self.layer1(x))) + x
        #return self.relu(x)
        return x

class Generator(keras.Model):
    def __init__(self, n_layers, block_dim):
        super(Generator, self).__init__()

        self.ff = keras.layers.Dense(block_dim, activation='linear')
        self.n_layers = n_layers
        self.res_blocks = [ResNetBlock(block_dim) for _ in range(n_layers)]
        self.relu = keras.layers.ReLU()

    def call(self, x):
        x = self.ff(x)
        for i in range(self.n_layers):
            x = self.res_blocks[i](x)
        #return self.relu(x)
        return x

class Discriminator(keras.Model):
    def __init__(self, n_layers, block_dim):
        super(Discriminator, self).__init__()

        self.n_layers = n_layers
        self.res_blocks = [ResNetBlock(block_dim) for _ in range(n_layers)]
        #self.res_blocks = [ResNetLeakyRelu(block_dim) for _ in range(n_layers)]

        self.relu = keras.layers.ReLU()
        self.ff1 = keras.layers.Dense(block_dim, activation='linear')
        self.ff2 = keras.layers.Dense(1, activation='linear')

    def call(self, x):
        x = self.ff1(x)
        for i in range(self.n_layers):
            x = self.res_blocks[i](x)
        #print("x before: {}".format(x)) 
        #x = self.relu(x)
        x = self.ff2(x)
        #print("x after: {}".format(x))
        #x = tf.nn.sigmoid(x)
        #print("x after sigmoid: {}".format(x))
        return x

if __name__ == '__main__':
    # create a generator
    generator = Generator(100, 20)
    
    # produce output using normal noise
    noise = np.random.normal(size = (1, 100))
    z = generator(noise[None, :, :])

    print(z)
