import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow as tf

import autoencoder
from gan import Generator, Discriminator

## get autoencoder parameters
with open('./autoencoder_parameters.json') as ae_file:
  ae_parameters = json.load(ae_file)
ae_data_parameters = ae_parameters['data_parameters']
ae_training_parameters = ae_parameters['training_parameters']
ae_architecture_parameters = ae_parameters['architecture_parameters']
ae_model_save_parameters = ae_parameters['model_save_parameters']

## get gan parameters
with open('./gan_parameters.json') as gan_file:
  gan_parameters = json.load(gan_file)
gan_training_parameters = gan_parameters['training_parameters']
gan_architecture_parameters = gan_parameters['architecture_parameters']
gan_model_save_parameters = gan_parameters['model_save_parameters']

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

# real_data shape == fake_data shape (batch_size, units)
def grad_penalty(real_data, fake_data, batch_size, discriminator):
    alpha = tf.random.uniform(shape = (batch_size, 1), minval = 0, maxval = 1)
    vect = alpha*real_data + (1-alpha)*fake_data
    with tf.GradientTape() as tape:
        # prediction shape: (batch_size, 1)
        tape.watch(vect)
        prediction = discriminator(vect)
    # gradients shape: (batch_size, num_variables) ?
    gradients = tape.gradient(prediction, vect)
    #grad_norm = tf.linalg.norm(gradients, axis = 1)
    #return tf.math.reduce_mean((grad_norm - 1)**2)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return gradient_penalty

def discriminator_loss(real_pred, fake_pred):
    ## Wasserstein loss -- no log here
    return -tf.math.reduce_mean(real_pred) + tf.math.reduce_mean(fake_pred)

def generator_loss(fake_pred):
    return -tf.math.reduce_mean(fake_pred)

def train_step_disc(real_data_batch, encoder, batch_size, generator, discriminator, optimizer):
    disc_loss = 0
    with tf.GradientTape() as tape:
        units = ae_architecture_parameters['units']
        z = tf.random.normal((batch_size, units*2))
        hidden = encoder.initialize_hidden_state()
        _, real_vects = encoder(real_data_batch, hidden)
        ## real_vects shape: (batch_size, units)

        #print("real: {}".format(real_vects))
        # discriminator for wgan actually predicts "level of realness" of image
        real_predictions = discriminator(real_vects)
        fake_vects = generator(z)
        #print("fake: {}".format(fake_vects))
        fake_predictions = discriminator(fake_vects)
        disc_loss = discriminator_loss(real_predictions, fake_predictions)
        # 10 has worked best so far here
        disc_loss += 10*grad_penalty(real_vects, fake_vects, batch_size, discriminator)
    variables = discriminator.trainable_variables
    gradients = tape.gradient(disc_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return disc_loss

def train_step_gen(batch_size, generator, discriminator, optimizer):
    gen_loss = 0
    with tf.GradientTape() as tape:
        units = ae_architecture_parameters['units']
        z = tf.random.normal((batch_size, units*2))
        fake_predictions = discriminator(generator(z))
        gen_loss = generator_loss(fake_predictions)

    variables = generator.trainable_variables
    gradients = tape.gradient(gen_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return gen_loss

def decode_sentence(decoder, enc_hidden, tokenizer):
    result = ''
    dec_hidden = enc_hidden
    dec_hidden = [dec_hidden for _ in range(decoder.num_layers)]
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    for t in range(50):
        #predictions, dec_hidden = decoder(dec_input,
        #                                  dec_hidden, output)
        prediction, dec_hidden = decoder(dec_input, dec_hidden)

        predicted_id = tf.argmax(prediction[0]).numpy()

        result += tokenizer.index_word[predicted_id] + ' '

        dec_input = tf.expand_dims([predicted_id], 0)
        if tokenizer.index_word[predicted_id] == '<end>':
            break

    print(result)

def train_gan(train_set, generator, discriminator, encoder, decoder, tokenizer, disc_optimizer, gen_optimizer, batch_size, steps_per_epoch):

    # define num epochs
    num_epochs = gan_training_parameters['epochs']
    # define generator training frequency
    n_generator_train = gan_training_parameters['n_generator_train']
    # define ae architecture
    units = ae_architecture_parameters['units']

    # define checkpoints
    gan_checkpoint_dir = gan_model_save_parameters['checkpoint_directory']
    gan_checkpoint = tf.train.Checkpoint(disc_optimizer=disc_optimizer,
                                         gen_optimizer=gen_optimizer,
                                         generator=generator,
                                         discriminator=discriminator)
    max_to_keep = gan_model_save_parameters['max_to_keep']
    gan_manager = tf.train.CheckpointManager(gan_checkpoint, gan_checkpoint_dir, 
                                             max_to_keep=max_to_keep,
                                             checkpoint_name = "ckpt")

    # model will be saved after this many epochs
    save_freq = gan_model_save_parameters['save_frequency']

    # current epoch
    curr_epoch = 0
    if (gan_model_save_parameters['restore_model'] and os.path.isdir(gan_checkpoint_dir)):
        latest_checkpoint = tf.train.latest_checkpoint(gan_checkpoint_dir)
        curr_epoch = int(latest_checkpoint.split('-')[-1]) * save_freq
        curr_epoch = curr_epoch if curr_epoch <= num_epochs else num_epochs

        # restart training if already at num_epochs
        if curr_epoch < num_epochs:
            gan_checkpoint.restore(latest_checkpoint)
            print('GAN restored from checkpoint at epoch {}.'.format(curr_epoch))

    ## training steps
    with open(gan_model_save_parameters['losses_filename'], 'a+', newline='') as losses_file:
        writer = csv.writer(losses_file)
        writer.writerow(["Epoch", "Batch", "Disc_Loss", "Gen_Loss"])
    for epoch in range(curr_epoch, gan_training_parameters['epochs']):
        print("epoch {}".format(epoch+1))
        disc_loss = 0
        gen_loss = 0
        for (i, (x, y)) in enumerate(train_set.take(steps_per_epoch)):
            batch_disc_loss = train_step_disc(x, encoder, batch_size, generator, discriminator, disc_optimizer)
            disc_loss += batch_disc_loss
            if (i % n_generator_train) == 0:
                batch_gen_loss = train_step_gen(batch_size, generator, discriminator, gen_optimizer)
                gen_loss += batch_gen_loss
            if (i % 100) == 0:
                print("batch {}".format(i))
                print("Discriminator Loss: {}".format(batch_disc_loss))
                print("Generator Loss: {}".format(batch_gen_loss))
                with open(gan_model_save_parameters['losses_filename'], 'a', newline='') as losses_file:
                    writer = csv.writer(losses_file)
                    writer.writerow([epoch, i, batch_disc_loss.numpy(), batch_gen_loss.numpy()])
        disc_loss /= (i+1) 
        gen_loss /= (np.floor((i+1)/n_generator_train))
        print("Epoch Average Discriminator Loss: {}".format(disc_loss))
        print("Epoch Average Generator Loss: {}".format(gen_loss))

        if (epoch + 1) % save_freq == 0:
            gan_manager.save()
        
        for _ in range(5):
            decode_sentence(decoder, generator(tf.random.normal((1, units*2))), tokenizer)

        # saving (checkpoint) the model every save_freq epochs
        if (epoch + 1) % save_freq == 0:
            gan_manager.save()

    return gan_checkpoint

def main(train_data):
    ## rebuild autoencoder from checkpoint
    # create training set
    
    #input_tensor_train, target_tensor_train, _ = autoencoder.load_dataset(train_data, gan_training_parameters['num_train_examples'])
    #max_length = target_tensor_train.shape[1]
    
    # get tokenizer
    with open(ae_model_save_parameters['tokenizer_filename'], 'rb') as handle:
        tokenizer = pickle.load(handle)
    print('Tokenizer loaded from: {}'.format(ae_model_save_parameters['tokenizer_filename']))
    vocab_size = len(tokenizer.word_index)+1

    ''' added below @ tassos please confirm that this is correct '''
    max_length = ae_data_parameters['max_sentence_length']
    train = autoencoder.create_dataset(train_data, gan_training_parameters['num_train_examples'])
    input_tensor_train = tokenizer.texts_to_sequences(train)
    input_tensor_train = tf.keras.preprocessing.sequence.pad_sequences(input_tensor_train, padding='post',
                         truncating = 'post', maxlen = max_length, value=0)
    target_tensor_train = input_tensor_train
    ''' end of additions '''

    # load model from checkpoint
    learning_rate = gan_training_parameters['learning_rate']
    weight_decay = gan_training_parameters['weight_decay']
    beta_1 = gan_training_parameters['beta_1']
    beta_2 = gan_training_parameters['beta_2']
    ae_optimizer = eval(ae_training_parameters['optimizer'])
    encoder = autoencoder.Encoder(vocab_size)
    decoder = autoencoder.Decoder(vocab_size)
    checkpoint = tf.train.Checkpoint(optimizer=ae_optimizer,
                                        encoder=encoder,
                                        decoder=decoder)
    checkpoint_dir = ae_model_save_parameters['checkpoint_directory']
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_checkpoint)
    print("AE restored from: {}".format(tf.train.latest_checkpoint(checkpoint_dir)))

    ## define variables for training
    # define batches
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = gan_training_parameters['batch_size']
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    # define autoencoder architecture
    units = ae_architecture_parameters['units']

    ## create datasets
    train_set = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    train_set = train_set.batch(BATCH_SIZE, drop_remainder=True)

    ## construct model
    gen_layers = gan_architecture_parameters['generator_num_layers']
    disc_layers = gan_architecture_parameters['discriminator_num_layers']
    generator = Generator(gen_layers, units*2)
    discriminator = Discriminator(disc_layers, units*2)

    ## define optimizers
    learning_rate = gan_training_parameters['learning_rate']
    weight_decay = gan_training_parameters['weight_decay']
    beta_1 = gan_training_parameters['beta_1']
    beta_2 = gan_training_parameters['beta_2']
    disc_optimizer = eval(gan_training_parameters['optimizer'])
    gen_optimizer = eval(gan_training_parameters['optimizer'])

    ## train GAN
    gan_checkpoint = train_gan(train_set, generator, discriminator, encoder, decoder, tokenizer, disc_optimizer, gen_optimizer, BATCH_SIZE, steps_per_epoch)

    return gan_checkpoint

if __name__ == '__main__':
    # Add input arguments
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train dataset, one sentence per line.")

    args = parser.parse_args()
    train_data = args.train_data

    main(train_data)
