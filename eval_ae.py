import tensorflow as tf
from autoencoder_reg import Encoder, Decoder, preprocess_sentence, load_dataset, create_dataset


def eval_sentence(sentence, tokenizer, encoder, decoder):
    sentence = preprocess_sentence(sentence)
    #inputs = [tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tokenizer.texts_to_sequences([sentence])[0]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                        #maxlen=max_length_inp,
                                                        maxlen = 50,
                                                        padding='post')
    inputs = tf.convert_to_tensor(inputs)

    #result = ''

    hidden = encoder.initialize_hidden_state()
    #hidden = tf.zeros((1, units))
    #enc_hidden = encoder(inputs, hidden)
    output, enc_hidden = encoder(inputs, hidden)
    decode_sentence(decoder, enc_hidden, tokenizer)


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

def main():
    ## for replication and model restoration
    #np.random.seed(1234)

    ## define variables for preprocessing
    # maximum length of sentences
    max_sentence_len = 40
    # maximum number of words in vocabulary
    max_vocab = 20000
    # number of examples (sentences) to use
    #num_train_examples = 200000
    num_train_examples = 150000
    num_dev_examples = 2000

    ## create datasets
    input_tensor_train, target_tensor_train, tokenizer = load_dataset("train.txt", max_sentence_len, max_vocab, num_train_examples)
    #dev = create_dataset(dev_data, num_dev_examples)
    #input_tensor_dev = tokenizer.texts_to_sequences(dev)
    #target_tensor_dev = input_tensor_dev
    #input_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(input_tensor_dev, padding='post',
    #                                                        truncating = 'post', maxlen = max_sentence_len, value=0)
    #target_tensor_dev = tf.keras.preprocessing.sequence.pad_sequences(target_tensor_dev, padding='post',
    #                                                        truncating = 'post', maxlen = max_sentence_len, value=0)

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
    # define optimizer (Adam)
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = "./training_ckpt_2"

    ## create datasets
    train_set = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    train_set = train_set.batch(BATCH_SIZE, drop_remainder=True)

    #dev_set = tf.data.Dataset.from_tensor_slices((input_tensor_dev, target_tensor_dev))
    #dev_batch = dev_set.batch(num_dev_examples)

    ## construct model
    encoder = Encoder(BATCH_SIZE, vocab_size, embedding_dim, units, 1)
    #decoder = Decoder(vocab_size, embedding_dim, units*2, 1)
    decoder = Decoder(vocab_size, embedding_dim, units*2, 1)


    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=encoder,
                                    decoder=decoder)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_checkpoint)

    sentences  = ["It is very cold today.", "I come from a large family of three brothers and three sisters.",
                  "How are you feeling today?", "He had to cancel his plans because he was not feeling well."]

    for s in sentences:
        print(s)
        eval_sentence(s, tokenizer, encoder, decoder)

if __name__ == '__main__':
    main()
