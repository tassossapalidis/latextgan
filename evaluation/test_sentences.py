import argparse
import json
import random
from nltk.translate.bleu_score import sentence_bleu
import pickle
import sys
import tensorflow as tf
import warnings

import autoencoder

########### Input number of random dev sentences to test here ###########
num_sentences = 2000
#########################################################################

########### Select False if evaluating entire dev set ###################
print_results = False
#########################################################################

## get parameters
with open('./autoencoder_parameters.json') as f:
  parameters = json.load(f)
data_parameters = parameters['data_parameters']
training_parameters = parameters['training_parameters']
architecture_parameters = parameters['architecture_parameters']
model_save_parameters = parameters['model_save_parameters']

def main(train_data, dev_data):
    with open(dev_data) as file:
        lines = list(file.read().splitlines())
    lines = [x for x in lines if len(x.split(' '))<= data_parameters['max_sentence_length']]

    test_sentences = []
    for i in range(num_sentences):
        test_sentences += [random.choice(lines)]
    
    ## Create vocabulary
    with open(model_save_parameters['tokenizer_filename'], 'rb') as handle:
        tokenizer = pickle.load(handle)
    print('Tokenizer loaded from: {}'.format(model_save_parameters['tokenizer_filename']))
    vocab_size = len(tokenizer.word_index)+1

    ## load model from checkpoint
    learning_rate = training_parameters['learning_rate']
    weight_decay = training_parameters['weight_decay'] # if using AdamW
    beta_1 = training_parameters['beta_1']
    beta_2 = training_parameters['beta_2']
    optimizer = eval(training_parameters['optimizer'])
    encoder = autoencoder.Encoder(vocab_size)
    decoder = autoencoder.Decoder(vocab_size)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint_dir = model_save_parameters['checkpoint_directory']
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest_checkpoint)

    # get parameters
    max_sentence_length = data_parameters['max_sentence_length']
    units = architecture_parameters['units']
    
    # get average BLEU
    BLEU1_avg = 0
    BLEU2_avg = 0
    BLEU3_avg = 0
    BLEU4_avg = 0

    for sentence in test_sentences:
        sentence = autoencoder.preprocess_sentence(sentence)
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
        
        if print_results == True:
            print('Original Sentence: {}'.format(sentence))
            print('Output Sentence:   <start> {}'.format(result))

        # compute BLEU score
        sentence = sentence.split(" ")
        result = result.split(" ")
        result = tokenizer.texts_to_sequences(result)
        result = ['<start>'] + tokenizer.sequences_to_texts(result)
        result = result[:-1]

        weights1 = (1, 0, 0, 0)
        weights2 = (1/2, 1/2, 0, 0)
        weights3 = (1/3, 1/3, 1/3, 0)
        weights4 = (1/4, 1/4, 1/4, 1/4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            BLEU1 = sentence_bleu([sentence], result, weights = weights1)
            BLEU2 = sentence_bleu([sentence], result, weights = weights2)
            BLEU3 = sentence_bleu([sentence], result, weights = weights3)
            BLEU4 = sentence_bleu([sentence], result, weights = weights4)
            
        BLEU1_avg += BLEU1
        BLEU2_avg += BLEU2
        BLEU3_avg += BLEU3
        BLEU4_avg += BLEU4
        
        if print_results == True:
            print('BLEU-1: {}'.format(BLEU1))
            print('BLEU-2: {}'.format(BLEU2))
            print('BLEU-3: {}'.format(BLEU3))
            print('BLEU-4: {}'.format(BLEU4))
            print('')

    BLEU1_avg /= num_sentences
    BLEU2_avg /= num_sentences
    BLEU3_avg /= num_sentences
    BLEU4_avg /= num_sentences
    
    print('Average BLEU-1: {}'.format(BLEU1_avg))
    print('Average BLEU-2: {}'.format(BLEU2_avg))
    print('Average BLEU-3: {}'.format(BLEU3_avg))
    print('Average BLEU-4: {}'.format(BLEU4_avg))
    
if __name__ == '__main__':
    # Add input arguments
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train dataset, one sentence per line.")
    parser.add_argument("--dev_data", type=str, help="path to dev dataset, one sentence per line.")

    args = parser.parse_args()
    train_data = args.train_data
    dev_data = args.dev_data

    main(train_data, dev_data)
    