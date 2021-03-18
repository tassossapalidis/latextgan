import argparse
import json
import random
from nltk.translate.bleu_score import sentence_bleu
from progressbar import ProgressBar
import re
import sys
import tensorflow as tf
import warnings

########### Select False if evaluating long sentence set ################
print_results = False
#########################################################################

## get parameters
with open('/content/drive/MyDrive/CS230project/model_parameter_tests/workfolder_tassos/autoencoder_parameters.json') as ae_file:
  ae_parameters = json.load(ae_file)
ae_data_parameters = ae_parameters['data_parameters']

with open('/content/drive/MyDrive/CS230project/model_parameter_tests/workfolder_tassos/gan_parameters.json') as gan_file:
  gan_parameters = json.load(gan_file)
gan_training_parameters = gan_parameters['training_parameters']

def main(train_data, sample_sentences):
    with open(train_data) as corpus_file:
        corpus_sentences = list(corpus_file.read().splitlines())
    
    corpus_sentences = [re.sub(r"(^\"|\"$)", "", x) for x in corpus_sentences]
    corpus_sentences = [x[:-1] + ' .' for x in corpus_sentences[:gan_training_parameters['num_train_examples']]]
    corpus_sentences = [x.split(" ") for x in corpus_sentences]
    corpus_sentences = [x for x in corpus_sentences if len(x) <= ae_data_parameters['max_sentence_length']]

    with open(sample_sentences) as sample_file:
        sample_sentence_list = list(sample_file.read().splitlines())
    
    sample_sentence_list = [x[:-1] for x in sample_sentence_list]
    sample_sentence_list = [x.split(" ")[:-1] for x in sample_sentence_list]
    
    # get average BLEU
    BLEU1_avg = 0
    BLEU2_avg = 0
    BLEU3_avg = 0
    BLEU4_avg = 0

    # define BLEU weights
    weights1 = (1, 0, 0, 0)
    weights2 = (1/2, 1/2, 0, 0)
    weights3 = (1/3, 1/3, 1/3, 0)
    weights4 = (1/4, 1/4, 1/4, 1/4)

    # compute BLEU score
    pbar = ProgressBar()
    for sentence in pbar(sample_sentence_list):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            BLEU1 = sentence_bleu(corpus_sentences, sentence, weights = weights1)
            BLEU2 = sentence_bleu(corpus_sentences, sentence, weights = weights2)
            BLEU3 = sentence_bleu(corpus_sentences, sentence, weights = weights3)
            BLEU4 = sentence_bleu(corpus_sentences, sentence, weights = weights4)
            
            
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

    BLEU1_avg /= len(sample_sentence_list)
    BLEU2_avg /= len(sample_sentence_list)
    BLEU3_avg /= len(sample_sentence_list)
    BLEU4_avg /= len(sample_sentence_list)

    print('Average BLEU-1: {}'.format(BLEU1_avg))
    print('Average BLEU-2: {}'.format(BLEU2_avg))
    print('Average BLEU-3: {}'.format(BLEU3_avg))
    print('Average BLEU-4: {}'.format(BLEU4_avg))
    
if __name__ == '__main__':
    # Add input arguments
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train dataset, one sentence per line.")
    parser.add_argument("--sample_sentences", type=str, help="path to sample sentences, one sentence per line.")

    args = parser.parse_args()
    train_data = args.train_data
    sample_sentences = args.sample_sentences

    main(train_data, sample_sentences)
    