import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

MAX_SEQUENCE_LENGTH = 29

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer

def process_data(data):
    ## '<' and '>' will be start and end tokens
    data = ['<' + l + '>' for l in data]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True) 
    tokenizer.fit_on_texts(data)
    tensor = tokenizer.texts_to_sequences(data)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen = MAX_SEQUENCE_LENGTH+2)
    return tensor, tokenizer


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, num_layers, batch_size, rate=0.1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        #self.enc_layers = [tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1024, return_state=True, return_sequences=True)) for _ in range(num_layers)]
        #self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(1024, return_sequences=True, return_state=True))
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(512, return_sequences=True, return_state=True))
        self.batch_size = batch_size

    def call(self, x, hidden):
        x = self.embedding(x)
      
        #output, hidden_state = self.gru(x, initial_state=hidden)
        output, hidden_state_forward, hidden_state_backward = self.gru(x, initial_state=hidden)
        '''for i in range(self.num_layers): 
            if i == 0:
                x, hidden_state_forward, hidden_state_backward = self.enc_layers[i](x, initial_state=hidden)
            else:
                x, hidden_state_forward, hidden_state_backward = self.enc_layers[i](x)
            hidden = [hidden_state_forward, hidden_state_backward]'''
        #return output, hidden_state_forward
        return output, tf.concat([hidden_state_forward, hidden_state_backward], axis=-1)

    def initialize_hidden_state(self):
        #return tf.zeros((self.batch_size, 1024))
        ## hidden state for bidirectional RNN
        return (tf.zeros((self.batch_size, 512)), tf.zeros((self.batch_size, 512)))

def attention(q, k, v):
    ### Transformer self attention
    ## values: (batch_size, num_heads, max_seq_len, depth), query: (batch_size, num_heads, 1, units)
    # query_exp = tf.expand_dims(query, 1)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    score = tf.linalg.matmul(q, k, transpose_b=True)/tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(score, axis=-1)
    context = tf.matmul(attention_weights, v)
    
    return context, attention_weights
   
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, units, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.units = units
        self.num_heads = num_heads

        assert units % num_heads == 0

        self.depth = units // num_heads

        self.wq = tf.keras.layers.Dense(units)
        self.wk = tf.keras.layers.Dense(units)
        self.wv = tf.keras.layers.Dense(units)

        self.dense = tf.keras.layers.Dense(units)

    def split_heads(self, x, batch_size):
        # x shape == (batch_size, seq_len, units)
        # in this case, seq_len == 1 so make sure to expand_dims before
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # after reshape, x shape == (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # attn_weights shape == (batch_size, num_heads, max_seq_len)
        # concat_attn_shape == (batch_size, units) 
        attn, attn_weights = attention(q, k, v)
        attn = tf.transpose(attn, perm=[0,2,1,3])
        attn = tf.squeeze(attn, axis=1)
        concat_attn = tf.reshape(attn, [batch_size, self.units])
        output = self.dense(concat_attn)

        return output, attn_weights
        

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, num_layers, rate=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dec_layer = tf.keras.layers.GRU(1024, return_sequences=True, return_state=True)
        #self.dec_layers = [tf.keras.layers.GRU(1024, return_sequences=True, return_state=True) for _ in range(num_layers)]
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.attention = MultiHeadAttention(1024, 2)

    def call(self, x, hidden, enc_output):
        context, weights = self.attention(hidden, enc_output, enc_output)
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state = self.dec_layer(x, initial_state=hidden)
        output = tf.squeeze(output, axis=1)
        output = self.dense(output)
        return output, state, weights

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_fn(real, pred):
    ## padding mask
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    
    loss_ = loss_obj(real, pred)
    mask = tf.squeeze(tf.cast(mask, dtype=loss_.dtype))
    loss_ *= mask
  
    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')

#### MAIN ####
factors, expansions = load_file("train.txt")
factors, fac_tokenizer = process_data(factors)
expansions, exp_tokenizer = process_data(expansions)

## +1 is for 0 padding
input_vocab = tf.reduce_max(factors) + 1
target_vocab = tf.reduce_max(expansions) + 1

BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((factors[:300000], expansions[:300000]))
train_data = dataset.shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(input_vocab, 128, 2, 128)
decoder = Decoder(target_vocab, 128, 2)

optimizer = tf.keras.optimizers.Adam()

checkpoint_path = "./checkpoints_multihead_2_300000_train/train"

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

#@tf.function(input_signature=train_step_signature)
def train_step(inp, tar, enc_hidden):
    loss = 0
    start_token = exp_tokenizer.texts_to_sequences(['<'])[0]

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        ## initial decoder hidden layer is output of encoder hidden layer
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(start_token*BATCH_SIZE, 1)

        for t in range(1, tar.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_fn(tar[:, t], predictions)
            # teacher forcing
            dec_input = tf.expand_dims(tar[:, t], 1)
        #print(loss)
        batch_loss = (loss / int(tar.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss


for epoch in range(2):
    train_loss.reset_states()
    #train_accuracy.reset_states()

    enc_hidden = encoder.initialize_hidden_state()
    for (batch, (inp, tar)) in enumerate(train_data):
        loss = train_step(inp, tar, enc_hidden)
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss.numpy()))

    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

        print ('Epoch {} Loss {:.4f}'.format(epoch + 1, 
                                                train_loss.result()))


start_token = exp_tokenizer.texts_to_sequences(['<'])[0]
end_token = exp_tokenizer.texts_to_sequences(['>'])[0][0]
attention_across_time = tf.zeros((1))
for j in range(500099, 500100):
    i = tf.expand_dims(factors[j], 0)
    #hidden = [tf.zeros((1, 1024))]*2
    hidden = [tf.zeros((1, 512))]*2
    enc_output, enc_hidden = encoder(i, hidden)
    dec_hidden = enc_hidden
    dec_output = tf.expand_dims(start_token, 0)
    result = ''
    for t in range(MAX_SEQUENCE_LENGTH):
        predictions, dec_hidden, attention_weights = decoder(dec_output, dec_hidden, enc_output)
        attention_weights = tf.squeeze(attention_weights, axis=0)
        if (len(tf.shape(attention_across_time)) == 1):
            attention_across_time = attention_weights
        else:
            attention_across_time = tf.concat((attention_across_time, attention_weights), axis = 1)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if predicted_id == end_token:
            break
        result += exp_tokenizer.sequences_to_texts([[predicted_id]])[0][0]
        dec_output = tf.expand_dims([predicted_id], 0)
    print("{} and {}".format(result, exp_tokenizer.sequences_to_texts([expansions[j]])))


def plot_attention_weights(attention, sentence, result):
    fig = plt.figure(figsize=[6.4,6.4])

    #attention = tf.squeeze(attention, axis=0)
    sentence = sentence[0].split()
    print(sentence)

    for head in range(attention.shape[0]):
        #ax = plt.subplot(1,attention.shape[0],head+1)
        ax = fig.add_subplot(2, attention.shape[0]/2, head+1)

        # plot the attention weights
        ax.matshow(attention[head], cmap='viridis')

        fontdict = {'fontsize': 9}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict)
        ax.set_yticklabels([''] + [i for i in result], fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))


    plt.tight_layout()
    plt.show()

plot_attention_weights(attention_across_time, fac_tokenizer.sequences_to_texts([factors[500099]]), result)

