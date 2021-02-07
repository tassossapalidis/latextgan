import numpy as np
import tensorflow as tf
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import os
import io

## for replication and model restoration
np.random.seed(1234)

# all preprocessing (and much of model) taken from https://www.tensorflow.org/tutorials/text/nmt_with_attention
# but instead of translating english -> spanish, model just translates english -> english

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# testing with using all training examples
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    #word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    
    ## just english part
    ## randomly sample num_examples from the dataset
    lines = np.random.choice(lines, size=num_examples, replace=False)
    word_pairs = [[preprocess_sentence(l.split('\t')[0]) for w in l.split('\t')] for l in lines]

    return zip(*word_pairs)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)
    
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = 100000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.05)
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
#embedding_dim = 128
embedding_dim = 256
#units = 1024
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

## +1 for zero padding
vocab_size = len(inp_lang.word_index)+1


class Encoder(tf.keras.Model):
    def __init__(self, batch_size, vocab_size, embedding_dim, num_units):
        super(Encoder, self).__init__()
        self.vs = vocab_size
        self.bs = batch_size
        self.emb_dim = embedding_dim
        self.hidden_dim = num_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.hidden_dim, 
                                       return_sequences = True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')
        # bidirectional worked better
        # self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.hidden_dim, 
        #                                return_sequences = False,
        #                                return_state = True,
        #                                recurrent_initializer = 'glorot_uniform'))
        #self.lstm = tf.keras.layers.LSTM(units)
    def call(self, x, hidden):
        # x shape after embedding : (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        #output, state_forward, state_backward = self.gru(x, initial_state = hidden)
        #output = self.lstm(x)
        # state shape: (batch_size, hidden_dim)

        #return state
        #return output, tf.concat([state_forward, state_backward], axis = 1)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.bs, self.hidden_dim))
        #return [tf.zeros((self.bs, self.hidden_dim)), tf.zeros((self.bs, self.hidden_dim))]

# Define attention mechanism
class BahdanauAttention(tf.keras.layers.Layer):
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

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_units):
        super(Decoder, self).__init__()
        self.vs = vocab_size
        self.emb_dim = embedding_dim
        self.hidden_dim = num_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.hidden_dim,
                                       return_sequences = False,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')
        #self.lstm = tf.keras.layers.LSTM(units, return_state=True)                               
        self.ff = tf.keras.layers.Dense(vocab_size)

        # attention
        self.attention = BahdanauAttention(self.hidden_dim)

    def call(self, x, hidden, enc_output):
        
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
        # output shape : (batch_size, 1, hidden_dim)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # output shape : (batch_size, 1, embedding_dim + hidden_size)

        output, state = self.gru(x, initial_state = hidden)
        #output, memory_state, carry_state = self.lstm(x, initial_state=hidden)

        output = self.ff(output)
        #return output, [memory_state, carry_state]
        return output, state

## loss function
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_fn(real, pred):
    ## padding mask
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_obj(real, pred)
    mask = tf.squeeze(tf.cast(mask, dtype=loss_.dtype))
    loss_ *= mask
 
    return tf.reduce_mean(loss_)

optimizer = tf.keras.optimizers.Adam()


## train model
encoder = Encoder(BATCH_SIZE, vocab_size, embedding_dim, units)
decoder = Decoder(vocab_size, embedding_dim, units)

# below is for bidirectional encoder
# decoder = Decoder(vocab_size, embedding_dim, units*2)
def train_step(inp, tar, hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, hidden)

        dec_hidden = enc_hidden
    
        ## dec_input.shape == (batch_size, 1), since only one word is being
        ## used as input at each timestep
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        #sentence = ''
        # teacher forcing
        for t in range(1, tar.shape[1]):
            prediction, dec_hidden = decoder(dec_input, dec_hidden, enc_output)

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

EPOCHS = 10
#checkpoint_dir = './training_checkpoints'
checkpoint_dir = "/content/drive/My Drive/Colab Notebooks/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
for epoch in range(EPOCHS):
    total_loss = 0

    hidden = encoder.initialize_hidden_state()
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)


def accuracy(prediction, label):
    return prediction == label

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
## evaluate performance on dev set
dev_set = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
loss = 0
acc = 0

for (x, y) in dev_set:
    x = tf.expand_dims(x, 0)
    y = tf.expand_dims(y, 0)
    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_hidden = encoder(x, hidden)
    
    dec_hidden = enc_hidden
   
    ## dec_input.shape == (batch_size, 1), since only one word is being
    ## used as input at each timestep
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    sentence = ''
    # teacher forcing
    #for t in range(1, x.shape[1]):
    t = 1
    predicted_id = targ_lang.word_index['<start>']
    while(predicted_id != targ_lang.word_index['<end>'] and t < y.shape[1]):    
        prediction, dec_hidden = decoder(dec_input, dec_hidden)

        loss += loss_fn(y[0, t], prediction)
        
        predicted_id = tf.argmax(prediction[0]).numpy()
        dec_input = tf.reshape(predicted_id, [1, 1])

        #if (tar[0, t].numpy() != 0) :       
        sentence += targ_lang.index_word[predicted_id] + " " 

        t += 1
    true_sentence = ''
    t = 1

    word_ind = targ_lang.word_index['<start>']
    while (word_ind != targ_lang.word_index['<end>']): 
        word_ind = y[0,t].numpy()
        true_sentence += targ_lang.index_word[word_ind] + " "
        t += 1
    acc += accuracy(sentence, true_sentence)

acc /= len(target_tensor_val)

# put the loss on the same scale as batch_loss
## average loss per word
loss /= (len(target_tensor_val)*int(y.shape[1]))
# this dev loss looks really high, but this really isn't a fair comparison
# because the training uses teacher focing while during prediction, we 
# cannot use teacher forcing. 
print("dev loss: {}".format(loss))
print("dev accuracy: {}".format(acc))

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
    
# model doesn't do well with compound sentences.
sentence = u'We went to class and then we went to the store.'
sentence = preprocess_sentence(sentence)
inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                       maxlen=max_length_inp,
                                                       padding='post')
inputs = tf.convert_to_tensor(inputs)

result = ''

hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
enc_hidden = encoder(inputs, hidden)

dec_hidden = enc_hidden
dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

for t in range(40):
    predictions, dec_hidden = decoder(dec_input,
                                      dec_hidden)

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    dec_input = tf.expand_dims([predicted_id], 0)
    if targ_lang.index_word[predicted_id] == '<end>':
        break
print(result)