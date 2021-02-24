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
        return self.relu(x)

class Generator(keras.Model):
    def __init__(self, n_layers, noise_dim, hidden_units, batch_size, max_sentence_length):
        super(Generator, self).__init__()

        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.bs = batch_size
        self.seq_len = max_sentence_length
        
        ## not sure if this will work but need to take input of size seq_len*units + noise_dim
        ## and condense it so output is of size units
        self.ff = keras.layers.Dense(hidden_units)
        self.res_blocks = [ResNetBlock(hidden_units) for _ in range(n_layers)]

    ## assume that previous output shape == (batch_size, max_seq_len, units)
    ## at each iteration, model will generate another word.
    def call(self, noise, previous_output):
        previous_output = tf.reshape(previous_output, [self.bs, -1])
        ## shape == (batch_size, max_seq_len*units)
        
        ## noise shape == (batch_size, 1)
        ## gen_input shape == (batch_size, seq_len*units+noise_dim)
        gen_input = tf.concat([noise, previous_output], axis = 1)
        
        x = self.ff(gen_input)
        for i in range(self.n_layers):
            x = self.res_blocks[i](x)

        return x;
        

class Discriminator(keras.Model):
    def __init__(self, n_layers, units):
        super(Discriminator, self).__init__()

        self.n_layers = n_layers
        self.res_blocks = [ResNetBlock(units) for _ in range(n_layers)]
        self.ff = keras.layers.Dense(1, activation='linear')

    def call(self, x):
        for i in range(self.n_layers):
            x = self.res_blocks[i](x)

        output = self.ff(x)
        return output

if __name__ == '__main__':
    # create a generator
    #generator = Generator(100, 20)
    generator = Generator(2, 16, 512, 64, 30)
    critic = Discriminator(2, 512)

    # produce output using normal noise
    noise = np.random.normal(size = (64, 16))
    fake_sentences = np.random.normal(size = (64, 30, 512))
    z = generator(noise, fake_sentences)

    print(z)
