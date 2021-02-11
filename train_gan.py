import os
import tensorflow as tf
import numpy as np
from autoencoder_general import Encoder, Decoder, load_dataset
from gan import Generator, Discriminator
## ALSO NEED AUTOENCODER MODEL
## since this model will be pre-trained, we need to load the weights in
## possibly just pass path to model.save() file

'''
assumptions:
- we have a working generator and discriminator
- generator takes a latent vector and outputs a context vector of 
  dimension equal to the context vectors generated by the autoencoder
- discriminator takes a context vector and outputs real number in (0 (fake), 1 (real))

- args: units -- size of hidden context vector (and, for now, also dimension of 
                 latent z being fed into generator)
        batch_size
        n_generator_train -- how frequently to train the generator
        epochs -- number of epochs to train for
'''

# call constructors
'''encoder = Encoder(args.units...)
decoder = Decoder(args.units...)
generator = ...
discriminator = ..'''

# make a checkpoint and restore autoencoder weights
# this might have to be saved_model; not sure
#checkpoint.restore_from_checkpoint(...)

units = 256
BATCH_SIZE = 64
vocab_size = 10000
embedding_dim = 256
BUFFER_SIZE = 1000
EPOCHS = 3
n_generator_train = 1

encoder = Encoder(BATCH_SIZE, vocab_size, embedding_dim, units)
decoder = Decoder(vocab_size, embedding_dim, units*2)
optimizer = tf.keras.optimizers.Adam()
generator = Generator(2, units*2)
discriminator = Discriminator(2, units*2)


input_tensor_train, target_tensor_train, tokenizer = load_dataset("Sentences.txt", 100, vocab_size, 500)
data = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
data = data.batch(BATCH_SIZE, drop_remainder=True)

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# real_data shape == fake_data shape (batch_size, units)
# TODO: check these shapes and this formula
def grad_penalty(real_data, fake_data):
    alpha = tf.random.uniform(shape = (BATCH_SIZE, 1), minval = 0, maxval = 1)
    vect = alpha*real_data + (1-alpha)*fake_data
    with tf.GradientTape() as tape:
        # prediction shape: (batch_size, 1)
        tape.watch(vect)
        prediction = discriminator(vect)
    # gradients shape: (batch_size, num_variables) ?
    gradients = tape.gradient(prediction, vect)
    grad_norm = tf.linalg.norm(gradients, axis = 1)
    return tf.math.reduce_mean((grad_norm - 1)**2)

def discriminator_loss(real_pred, fake_pred):
    ## Wasserstein loss -- no log here
    return -tf.math.reduce_mean(real_pred) + tf.math.reduce_mean(fake_pred)

def generator_loss(fake_pred):
    return -tf.math.reduce_mean(fake_pred)

def train_step_disc(real_data_batch):
    disc_loss = 0
    with tf.GradientTape() as tape:
        #z = tf.random.normal((args.batch_size, args.units))
        z = tf.random.normal((BATCH_SIZE, units*2))
        hidden = encoder.initialize_hidden_state()
        _, real_vects = encoder(real_data_batch, hidden)
        ## real_vects shape: (batch_size, units)
        
        # discriminator for wgan actually predicts "level of realness" of image
        real_predictions = discriminator(real_vects)
        fake_predictions = discriminator(generator(z))
        disc_loss = discriminator_loss(real_predictions, fake_predictions)
        variables = discriminator.trainable_variables
    gradients = tape.gradient(disc_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
        
    return disc_loss


def train_step_gen():
    gen_loss = 0 
    with tf.GradientTape() as tape:
        #z = tf.random.normal((args.batch_size, args.units))
        z = tf.random.normal((BATCH_SIZE, units*2))
        fake_predictions = discriminator(generator(z))
        gen_loss = generator_loss(fake_predictions)

        variables = generator.trainable_variables
    gradients = tape.gradient(gen_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return gen_loss

def main():
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(EPOCHS):
        print("epoch {}".format(epoch+1))
        disc_loss = 0
        for (i, (x, y)) in enumerate(data):
            disc_loss += train_step_disc(x)
        disc_loss /= i

        if (epoch % n_generator_train) == 0:
            gen_loss = 0

            # change this. don't actually need data but just want to make sure
            # generator is getting same number of updates as discriminator.
            # also not totally sure if this is the right thing to do 
            for i, _ in enumerate(data):
                gen_loss += train_step_gen()
            gen_loss /= i

if __name__ == '__main__':
    main()
