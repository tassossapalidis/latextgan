import numpy as np
import matplotlib.ticker as ticker
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf

import argparse
import json
import os
import re
import unicodedata

## get parameters
with open('./autoencoder_parameters.json') as f:
  parameters = json.load(f)
data_parameters = parameters['data_parameters']
training_parameters = parameters['training_parameters']
architecture_parameters = parameters['architecture_parameters']
model_save_parameters = parameters['model_save_parameters']
test_sentences = parameters['test_sentences']

# all preprocessing (and much of model) taken from https://www.tensorflow.org/tutorials/text/nmt_with_attention
# but instead of translating english -> spanish, model just translates english -> english

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# testing with using all training examples
def create_dataset(path, num_examples):
    max_sentence_len = data_parameters['max_sentence_length']
    with open(path) as f:
        lines = f.read().splitlines()

    ## input sentence == target sentence
    ## this is just to make sure dev set is different from train set for now... 
    ## once we actually split into train/dev/test sets we wont have to do this
    pre_processed_sentences = [l for l in lines if len(l.split(' '))<= max_sentence_len]
    processed_sentences = [preprocess_sentence(l) for l in pre_processed_sentences[:num_examples]]
    return processed_sentences


def tokenize(lang):
    max_sentence_len = data_parameters['max_sentence_length']
    max_vocab = data_parameters['max_vocab']
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = max_vocab, filters='', oov_token='<unk>')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    ## pad ragged tensor with zeros
    ## tensors longer than 'maxlen' will be truncated
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', truncating = 'post',
                                                           maxlen = max_sentence_len, value=0)

    return tensor, lang_tokenizer

def load_dataset(path, num_examples = None):
    # creating cleaned (input, output) pairs
    lang = create_dataset(path, num_examples)
    
    input_tensor, inp_lang_tokenizer = tokenize(lang)
    target_tensor = input_tensor

    return input_tensor, target_tensor, inp_lang_tokenizer

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.bs = training_parameters['batch_size']
        self.hidden_dim = architecture_parameters['units']
        self.num_layers = architecture_parameters['encoder_gru_layers']
        self.embedding_dim = architecture_parameters['embedding_dimension']
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        self.grus = [tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim, 
                                        return_sequences = True,
                                        return_state = True,
                                        recurrent_initializer = training_parameters['initializer'], 
                                        dropout = training_parameters['dropout'], 
                                        recurrent_dropout = training_parameters['recurrent_dropout']))
                                        for _ in range(self.num_layers)]

        #self.lstm = tf.keras.layers.LSTM(units)
    def call(self, x, hidden):
        # x shape after embedding : (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        _ = hidden[0]
        _ = hidden[1]

        for i in range(self.num_layers):
          x, state_forward, state_backward = self.grus[i](x)
          
        # get the last hidden state
        output = x[:, -1, :]
        return output, tf.concat([state_forward, state_backward], axis = 1)

    def initialize_hidden_state(self):
        return [tf.zeros((self.bs, self.hidden_dim)), tf.zeros((self.bs, self.hidden_dim))]

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.embedding_dim = architecture_parameters['embedding_dimension']
        self.hidden_dim = architecture_parameters['units'] * 2
        self.num_layers = architecture_parameters['decoder_gru_layers']
        self.vocab_size = vocab_size

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        self.grus = [tf.keras.layers.GRU(self.hidden_dim,
                                       return_sequences = True,
                                       return_state = True,
                                       recurrent_initializer = training_parameters['initializer'], 
                                       dropout = training_parameters['dropout'], 
                                       recurrent_dropout = training_parameters['recurrent_dropout'])
                                        for _ in range(self.num_layers)]                             
                             
        self.ff = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden):

        x = self.embedding(x)
        # output shape : (batch_size, 1, hidden_dim)

        ## try to initialize the hidden states of each layer
        x, hidden[0] = self.grus[0](x, initial_state = hidden[0])

        for i in range(1, self.num_layers):
          x, hidden[i] = self.grus[i](x, initial_state = hidden[i])
        
        output = tf.squeeze(x, 1)
        output = self.ff(output)

        return output, hidden

def loss_fn(real, pred):
    ## padding mask
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_obj = eval(training_parameters['loss_function'])
    loss_ = loss_obj(real, pred)
    mask = tf.squeeze(tf.cast(mask, dtype=loss_.dtype))
    loss_ *= mask
 
    return tf.reduce_mean(loss_)

def train_step(inp, tar, hidden, encoder, decoder, tokenizer, optimizer):
    batch_size = training_parameters['batch_size']

    loss = 0

    with tf.GradientTape() as tape:
        _, enc_hidden = encoder(inp, hidden)

        dec_hidden = enc_hidden
    
        # used as input at each timestep
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

        dec_hidden = [dec_hidden for _ in range(decoder.num_layers)]

        # teacher forcing
        for t in range(1, tar.shape[1]):
            #prediction, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            prediction, dec_hidden = decoder(dec_input, dec_hidden)

            loss += loss_fn(tar[:, t], prediction)
            dec_input = tf.expand_dims(tar[:, t], 1)
            
        # average loss over sequence length
        batch_loss = loss/int(tar.shape[1])

        # apply one step of optimization
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

def train_autoencoder(train_set, encoder, decoder, tokenizer, optimizer, steps_per_epoch, dev_set, num_dev_examples):

    num_epochs = training_parameters['epochs']

    # define checkpoints
    checkpoint_dir = model_save_parameters['checkpoint_directory']
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    max_to_keep = model_save_parameters['max_to_keep']
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, 
                                         max_to_keep=max_to_keep,
                                         checkpoint_name = "ckpt")

    # model will be saved after this many epochs
    save_freq = model_save_parameters['save_frequency']

    # current epoch
    curr_epoch = 0
    if (model_save_parameters['restore_model'] and os.path.isdir(checkpoint_dir)):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        curr_epoch = int(latest_checkpoint.split('-')[-1]) * save_freq
        curr_epoch = curr_epoch if curr_epoch <= num_epochs else num_epochs

        # restart training if already at num_epochs
        if curr_epoch < num_epochs:
            checkpoint.restore(latest_checkpoint)
            print('Model restored from checkpoint at epoch {}.'.format(curr_epoch))

    for epoch in range(curr_epoch, training_parameters['epochs']):
        total_loss = 0

        hidden = encoder.initialize_hidden_state()
        for (batch, (inp, targ)) in enumerate(train_set.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, hidden, encoder, decoder, tokenizer, optimizer)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every save_freq epochs
        if (epoch + 1) % save_freq == 0:
            manager.save()

        if (epoch + 1) % 1 == 0:
            test_dev(encoder, decoder, tokenizer, dev_set, num_dev_examples)
    
    #checkpoint.save(file_prefix = checkpoint_prefix)
    return checkpoint

def accuracy(prediction, label):
    return prediction == label

def test_dev(encoder, decoder, tokenizer, dev_set, num_dev_examples):

    for (inp, tar) in dev_set.take(1):

      loss = 0

      hidden = [tf.zeros((num_dev_examples, encoder.hidden_dim)), tf.zeros((num_dev_examples, encoder.hidden_dim))]
      _, enc_hidden = encoder(inp, hidden)

      dec_hidden = enc_hidden
      dec_hidden = [dec_hidden for _ in range(decoder.num_layers)]

      ## used as input at each timestep
      dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * num_dev_examples, 1)

      # teacher forcing
      for t in range(1, tar.shape[1]):
          prediction, dec_hidden = decoder(dec_input, dec_hidden)

          loss += loss_fn(tar[:, t], prediction)
          dec_input = tf.expand_dims(tar[:, t], 1)

      # average loss over sequence length
      batch_loss = loss/int(tar.shape[1])

      print("dev loss: {}".format(batch_loss))

      return batch_loss

def eval_accuracy(encoder, decoder, tokenizer, dev_set):

    num_dev_examples = data_parameters['num_dev_examples']

    acc = 0

    units = encoder.hidden_dim

    for (x, y) in dev_set:
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
        hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
        _, enc_hidden = encoder(x, hidden)

        dec_hidden = enc_hidden
        dec_hidden = [dec_hidden for _ in range(decoder.num_layers)]
    
        ## dec_input.shape == (batch_size, 1), since only one word is being
        ## used as input at each timestep
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

        sentence = ''

        # teacher forcing
        t = 1
        predicted_id = tokenizer.word_index['<start>']
        while(t < y.shape[1] and predicted_id != tokenizer.word_index['<end>']):    
            #prediction, dec_hidden = decoder(dec_input, dec_hidden, output)
            prediction, dec_hidden = decoder(dec_input, dec_hidden)
            
            predicted_id = tf.argmax(prediction[0]).numpy()
            dec_input = tf.reshape(predicted_id, [1, 1])
      
            sentence += tokenizer.index_word[predicted_id] + " " 

            t += 1
        true_sentence = ''
        t = 1

        word_ind = tokenizer.word_index['<start>']

        while (t < y.shape[1] and word_ind != tokenizer.word_index['<end>']): 
            word_ind = y[0,t].numpy()
            true_sentence += tokenizer.index_word[word_ind] + " "
            t += 1
        acc += accuracy(sentence, true_sentence)

    acc /= num_dev_examples
    print("dev accuracy: {}".format(acc))

    return acc
    
def evaluate_sentences(train_data, tokenizer = None):
    if tokenizer == None:
        _, _, tokenizer = load_dataset(train_data, data_parameters['num_train_examples'])
    vocab_size = len(tokenizer.word_index)+1

    ## load model from checkpoint    learning_rate = training_parameters['learning_rate']
    weight_decay = training_parameters['weight_decay'] # if using AdamW
    beta_1 = training_parameters['beta_1']
    beta_2 = training_parameters['beta_2']
    optimizer = eval(training_parameters['optimizer'])
    encoder = Encoder(vocab_size)
    decoder = Decoder(vocab_size)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint_dir = model_save_parameters['checkpoint_directory']
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_checkpoint)

    # get parameters
    max_sentence_length = data_parameters['max_sentence_length']
    units = architecture_parameters['units']

    for test_sentence in test_sentences:
        sentence = test_sentence
        sentence = preprocess_sentence(sentence)
        inputs = tokenizer.texts_to_sequences([sentence])[0]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen = max_sentence_length,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
        _, enc_hidden = encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_hidden = [dec_hidden for _ in range(decoder.num_layers)]
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

        for t in range(50):
            prediction, dec_hidden = decoder(dec_input, dec_hidden)
            predicted_id = tf.argmax(prediction[0]).numpy()

            result += tokenizer.index_word[predicted_id] + ' '

            dec_input = tf.expand_dims([predicted_id], 0)
            if tokenizer.index_word[predicted_id] == '<end>':
                break
        
        print('Original Sentence: {}'.format(sentence))
        print('Output Sentence:   <start> {} \n'.format(result))

def main(train_data, dev_data):

    ## get parameter values
    num_train_examples = data_parameters['num_train_examples']
    num_dev_examples = data_parameters['num_dev_examples']
    max_sentence_len = data_parameters['max_sentence_length']

    ## create datasets
    input_tensor_train, target_tensor_train, tokenizer = load_dataset(train_data, num_train_examples)
    with open(model_save_parameters['tokenizer_filename'], 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Tokenizer saved to: {}'.format(model_save_parameters['tokenizer_filename']))
    dev = create_dataset(dev_data, num_dev_examples)
    input_tensor_dev = tokenizer.texts_to_sequences(dev)
    target_tensor_dev = input_tensor_dev
    input_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(input_tensor_dev, padding='post',
                                                            truncating = 'post', maxlen = max_sentence_len, value=0)
    target_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(target_tensor_dev, padding='post',
                                                            truncating = 'post', maxlen = max_sentence_len, value=0)

    ## define variables for training
    BATCH_SIZE = training_parameters['batch_size']
    units = architecture_parameters['units']
    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    vocab_size = len(tokenizer.word_index)+1
    num_dev_examples = len(target_tensor_dev)


    ## create datasets
    train_set = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    train_set = train_set.batch(BATCH_SIZE, drop_remainder=True)

    dev_set = tf.data.Dataset.from_tensor_slices((input_tensor_dev, target_tensor_dev))
    dev_batch = dev_set.batch(num_dev_examples)

    ## construct model
    encoder = Encoder(vocab_size)
    decoder = Decoder(vocab_size)

    ## construct optimizer
    weight_decay = training_parameters['weight_decay'] # if using AdamW
    beta_1 = training_parameters['beta_1']
    beta_2 = training_parameters['beta_2']
    optimizer = eval(training_parameters['optimizer'])

    ## train model
    checkpoint = train_autoencoder(train_set, encoder, decoder, tokenizer, optimizer, steps_per_epoch, dev_batch, num_dev_examples)
    
    evaluate_sentences(train_data, tokenizer)

    return checkpoint

if __name__ == '__main__':
    # Add input arguments
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train dataset, one sentence per line.")
    parser.add_argument("--dev_data", type=str, help="path to dev dataset, one sentence per line.")

    args = parser.parse_args()
    train_data = args.train_data
    dev_data = args.dev_data

    main(train_data, dev_data)
