import numpy as np
import tensorflow as tf
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import argparse
import unicodedata
import re
import os
import io

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

    # remove everything except (a-z, A-Z, ".", "?", "!", ",", " ")
    #w = re.sub(r"[^a-zA-Z?.!,\s]+", "", w)
    
    # replace everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,]+", " ", w)
    

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# testing with using all training examples
def create_dataset(path, num_examples, max_sentence_len):
    with open(path, encoding = "utf-8") as f:
        lines = f.read().splitlines()

    ## input sentence == target sentence
    ## this is just to make sure dev set is different from train set for now... 
    ## once we actually split into train/dev/test sets we wont have to do this
    processed_sentences = [preprocess_sentence(l) for l in lines[:num_examples] if len(l.split(' '))<=max_sentence_len]
    return processed_sentences


def tokenize(lang, max_sentence_len, max_vocab):
    ## changed num_words here to only have a 10,000 word vocab
    ## I think words not known by tokenizer will just be skipped, which is probably undesirable
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = max_vocab, filters='', oov_token='<unk>')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)

    ## pad ragged tensor with zeros
    ## tensors longer than 'maxlen' will be truncated
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', truncating = 'post',
                                                           maxlen = max_sentence_len, value=0)

    return tensor, lang_tokenizer

def load_dataset(path, max_sentence_len, max_vocab, num_examples = None):
    # creating cleaned (input, output) pairs
    lang = create_dataset(path, num_examples, max_sentence_len)
    
    input_tensor, inp_lang_tokenizer = tokenize(lang, max_sentence_len, max_vocab)
    target_tensor = input_tensor

    return input_tensor, target_tensor, inp_lang_tokenizer

class Encoder(tf.keras.Model):
    def __init__(self, batch_size, vocab_size, embedding_dim, num_units, num_layers):
        super(Encoder, self).__init__()
        self.bs = batch_size
        self.hidden_dim = num_units
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        '''self.gru = tf.keras.layers.GRU(self.hidden_dim, 
                                       return_sequences = True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')'''
        # bidirectional worked better
        '''self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim, 
                                        return_sequences = False,
                                        return_state = True,
                                        recurrent_initializer = 'glorot_uniform'))'''
        self.grus = [tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim, 
                                        return_sequences = True,
                                        return_state = True,
                                        recurrent_initializer = 'glorot_uniform'))
                for _ in range(num_layers)]
        '''self.grus = [tf.keras.layers.GRU(self.hidden_dim, 
                                        return_sequences = True,
                                        return_state = True,
                                        recurrent_initializer = 'glorot_uniform')
                for _ in range(num_layers)]'''
        #self.lstm = tf.keras.layers.LSTM(units)
    def call(self, x, hidden):
        # x shape after embedding : (batch_size, seq_len, embedding_dim)
        print(x.shape)
        x = self.embedding(x)
        #output, state_forward, state_backward = self.gru(x, initial_state = hidden)
        hidden_forward = hidden[0]
        hidden_backward = hidden[1]

        for i in range(self.num_layers):
          x, state_forward, state_backward = self.grus[i](x)
          #x, state = self.grus[i](x)
          
        

        # get the last hidden state
        output = x[:, -1, :]
        #output = tf.reduce_sum(x, 1)
        return output, tf.concat([state_forward, state_backward], axis = 1)
        #return output, state

    def initialize_hidden_state(self):
        #return tf.zeros((self.bs, self.hidden_dim))
        return [tf.zeros((self.bs, self.hidden_dim)), tf.zeros((self.bs, self.hidden_dim))]

# Define attention mechanism
'''class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights'''

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_units, num_layers):
        super(Decoder, self).__init__()
        self.emb_dim = embedding_dim
        self.hidden_dim = num_units
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        '''self.gru = tf.keras.layers.GRU(self.hidden_dim,
                                       return_sequences = False,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')'''
        self.grus = [tf.keras.layers.GRU(self.hidden_dim,
                                       return_sequences = True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')
                    for _ in range(num_layers)]                             
        # self.lstm = tf.keras.layers.LSTM(units, return_state=True)                               
        self.ff = tf.keras.layers.Dense(vocab_size)

        # attention
        #self.attention = BahdanauAttention(self.hidden_dim)

    #def call(self, x, hidden, enc_output):
    def call(self, x, hidden):

        x = self.embedding(x)
        # output shape : (batch_size, 1, hidden_dim)

        # output, state = self.gru(x, initial_state = hidden)
        x, _ = self.grus[0](x, hidden)
        for i in range(1, self.num_layers):
          x, hidden = self.grus[i](x, hidden)
        
        output = tf.squeeze(x, 1)
        output = self.ff(output)
        
        #return output, state
        return output, hidden

def loss_fn(real, pred):
    ## padding mask
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_obj(real, pred)
    mask = tf.squeeze(tf.cast(mask, dtype=loss_.dtype))
    loss_ *= mask
 
    return tf.reduce_mean(loss_)

def train_step(inp, tar, hidden, encoder, decoder, tokenizer, optimizer, batch_size):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, hidden)

        dec_hidden = enc_hidden
    
        ## dec_input.shape == (batch_size, 1), since only one word is being
        ## used as input at each timestep
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

        #sentence = ''
        # teacher forcing
        for t in range(1, tar.shape[1]):
            #prediction, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            prediction, dec_hidden = decoder(dec_input, dec_hidden)

            loss += loss_fn(tar[:, t], prediction)
            dec_input = tf.expand_dims(tar[:, t], 1)

            #if (tar[0, t].numpy() != 0) :
            #    sentence += targ_lang.index_word[tar[0,t].numpy()] + " " 
            
        # average loss over sequence length
        batch_loss = loss/int(tar.shape[1])
        #print(sentence)

        variables = encoder.trainable_variables + decoder.trainable_variables
    
        ## apply one step of optimization
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

def train_autoencoder(train_set, encoder, decoder, optimizer, tokenizer, num_epochs, batch_size, steps_per_epoch, 
                      dev_set, num_dev_examples, restore_model=True):
    #checkpoint_dir = "./training_checkpoints"
    #checkpoint_dir = "/content/drive/MyDrive/CS230project/training_chekpoints"
    checkpoint_dir = "./training_ckpt_2"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    
    # model will be saved after this many epochs
    save_freq = 2
    # current epoch
    curr_epoch = 0
    if (restore_model and os.path.isdir(checkpoint_dir)):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        print(latest_checkpoint)
        checkpoint.restore(latest_checkpoint)
        
        curr_epoch = int(latest_checkpoint.split('-')[-1]) * save_freq
        curr_epoch = curr_epoch if curr_epoch <= num_epochs else num_epochs
        print('Model restored from checkpoint at epoch {}.'.format(curr_epoch))

    for epoch in range(curr_epoch, num_epochs):
        total_loss = 0

        hidden = encoder.initialize_hidden_state()
        for (batch, (inp, targ)) in enumerate(train_set.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, hidden, encoder, decoder, tokenizer, optimizer, batch_size)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % save_freq == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        if (epoch + 1) % 5 == 0:
            test_dev(encoder, decoder, tokenizer, dev_set, num_dev_examples)
    
    #checkpoint.save(file_prefix = checkpoint_prefix)
    return checkpoint

def accuracy(prediction, label):
    return prediction == label

def test_dev(encoder, decoder, tokenizer, dev_set, num_dev_examples):

    for (inp, tar) in dev_set.take(1):
      print(inp.shape)

      loss = 0

      hidden = [tf.zeros((num_dev_examples, encoder.hidden_dim)), tf.zeros((num_dev_examples, encoder.hidden_dim))]
      enc_output, enc_hidden = encoder(inp, hidden)

      dec_hidden = enc_hidden
    
      ## dec_input.shape == (batch_size, 1), since only one word is being
      ## used as input at each timestep
      dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * num_dev_examples, 1)

      # teacher forcing
      for t in range(1, tar.shape[1]):
          #prediction, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
          prediction, dec_hidden = decoder(dec_input, dec_hidden)

          loss += loss_fn(tar[:, t], prediction)
          dec_input = tf.expand_dims(tar[:, t], 1)

      # average loss over sequence length
      batch_loss = loss/int(tar.shape[1])

      print("dev loss: {}".format(batch_loss))

      return batch_loss

    '''loss = 0
    acc = 0

    units = encoder.hidden_dim

    for (x, y) in dev_set:
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
        hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
        #hidden = tf.zeros((1, units))
        #enc_hidden = encoder(x, hidden)
        output, enc_hidden = encoder(x, hidden)

        dec_hidden = enc_hidden
    
        ## dec_input.shape == (batch_size, 1), since only one word is being
        ## used as input at each timestep
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

        sentence = ''
        # teacher forcing
        #for t in range(1, x.shape[1]):
        t = 1
        predicted_id = tokenizer.word_index['<start>']
        while(predicted_id != tokenizer.word_index['<end>'] and t < y.shape[1]):    
            #prediction, dec_hidden = decoder(dec_input, dec_hidden, output)
            prediction, dec_hidden = decoder(dec_input, dec_hidden)

            loss += loss_fn(y[0, t], prediction)
            
            predicted_id = tf.argmax(prediction[0]).numpy()
            dec_input = tf.reshape(predicted_id, [1, 1])

            #if (tar[0, t].numpy() != 0) :       
            sentence += tokenizer.index_word[predicted_id] + " " 

            t += 1
        true_sentence = ''
        t = 1

        word_ind = tokenizer.word_index['<start>']
        while (word_ind != tokenizer.word_index['<end>']): 
            word_ind = y[0,t].numpy()
            true_sentence += tokenizer.index_word[word_ind] + " "
            t += 1
        acc += accuracy(sentence, true_sentence)

    acc /= num_dev_examples

    # put the loss on the same scale as batch_loss
    ## average loss per word
    loss /= (num_dev_examples*int(y.shape[1]))
    # this dev loss looks really high, but this really isn't a fair comparison
    # because the training uses teacher focing while during prediction, we 
    # cannot use teacher forcing. 
    print("dev loss: {}".format(loss))
    print("dev accuracy: {}".format(acc))

    return loss, acc'''

def eval_accuracy(encoder, decoder, tokenizer, dev_set, num_dev_examples):

    acc = 0

    units = encoder.hidden_dim

    for (x, y) in dev_set:
        x = tf.expand_dims(x, 0)
        y = tf.expand_dims(y, 0)
        hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
        #hidden = tf.zeros((1, units))
        #enc_hidden = encoder(x, hidden)
        output, enc_hidden = encoder(x, hidden)

        dec_hidden = enc_hidden
    
        ## dec_input.shape == (batch_size, 1), since only one word is being
        ## used as input at each timestep
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

        sentence = ''
        # teacher forcing
        #for t in range(1, x.shape[1]):
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

def main(train_data, dev_data, test_sentence):
    ## for replication and model restoration
    #np.random.seed(1234)

    ## define variables for preprocessing
    # maximum length of sentences
    max_sentence_len = 30
    # maximum number of words in vocabulary
    max_vocab = 20000
    # number of examples (sentences) to use
    #num_train_examples = 200000
    num_train_examples = 1500
    num_dev_examples = 200

    ## create datasets
    input_tensor_train, target_tensor_train, tokenizer = load_dataset(train_data, max_sentence_len, max_vocab, num_train_examples)
    dev = create_dataset(dev_data, num_dev_examples, max_sentence_len)
    input_tensor_dev = tokenizer.texts_to_sequences(dev)
    target_tensor_dev = input_tensor_dev
    input_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(input_tensor_dev, padding='post',
                                                            truncating = 'post', maxlen = max_sentence_len, value=0)
    target_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(target_tensor_dev, padding='post',
                                                            truncating = 'post', maxlen = max_sentence_len, value=0)

    ## define variables for training
    # Number of epochs
    EPOCHS = 12
    # restore from checkpoint file?
    restore_model = True
    # define batches
    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    # define autoencoder architecture
    embedding_dim = 128
    #units = 256
    units = 512
    # Calculate max_length of the tensors
    max_length = target_tensor_train.shape[1]
    # calculate vocab size (+1 for zero padding)
    vocab_size = len(tokenizer.word_index)+1
    # number of dev examples
    num_dev_examples = len(target_tensor_dev)
    # define optimizer (Adam)
    optimizer = tf.keras.optimizers.Adam()


    ## create datasets
    train_set = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    train_set = train_set.batch(BATCH_SIZE, drop_remainder=True)

    dev_set = tf.data.Dataset.from_tensor_slices((input_tensor_dev, target_tensor_dev))
    dev_batch = dev_set.batch(num_dev_examples)

    ## construct model
    encoder = Encoder(BATCH_SIZE, vocab_size, embedding_dim, units, 1)
    #decoder = Decoder(vocab_size, embedding_dim, units*2, 2)
    decoder = Decoder(vocab_size, embedding_dim, units*2, 1)

    ## train model
    checkpoint = train_autoencoder(train_set, encoder, decoder, optimizer, tokenizer, EPOCHS, BATCH_SIZE, 
                                   steps_per_epoch, dev_batch, num_dev_examples, restore_model)

    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    ## evaluate performance on dev set
    #test_dev(encoder, decoder, tokenizer, dev_batch, num_dev_examples)
    #eval_accuracy(encoder, decoder, tokenizer, dev_set, num_dev_examples)
        
    # model doesn't do well with compound sentences.
    sentence = test_sentence
    sentence = preprocess_sentence(sentence)
    #inputs = [tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tokenizer.texts_to_sequences([sentence])[0]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                        #maxlen=max_length_inp,
                                                        maxlen = 40,
                                                        padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    #hidden = tf.zeros((1, units))
    #enc_hidden = encoder(inputs, hidden)
    output, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
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
    return checkpoint

if __name__ == '__main__':
    # Add input arguments
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train dataset, one sentence per line.")
    parser.add_argument("--dev_data", type=str, help="path to dev dataset, one sentence per line.")
    ## might also need a max_length argument? Otherwise the dimensions of this model might not match
    ## the dimensions needed for the future (encoder?)? Not sure why this would be the case because 
    ## RNNs should be able to handle variable length input but
    args = parser.parse_args()
    train_data = args.train_data
    dev_data = args.dev_data

    test_sentence = "The air is cold in the winter."

    main(train_data, dev_data, test_sentence)

    # old code for reference
    '''seq_len = 0
    for (x, y) in dev_set.take(1):
    seq_len = x.shape[0]
    hidden = [tf.zeros((len(target_tensor_val), seq_len)), tf.zeros((len(target_tensor_val), seq_len))] 
    enc_hidden = encoder(dev_set[0], hidden)

    dec_hidden = enc_hidden
        
    ## dec_input.shape == (batch_size, 1), since only one word is being
    ## used as input at each timestep
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    #sentence = ''
    # teacher forcing
    for t in range(1, seq_len):
    prediction, dec_hidden = decoder(dec_input, dec_hidden)

    loss += loss_fn(tar[:, t], prediction)
    dec_input = tf.expand_dims(tar[:, t], 1)

    loss = loss/seq_len
    print("dev loss: {}".format(loss))'''