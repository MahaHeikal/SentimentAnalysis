# -*- coding: utf8 -*-

import numpy as np
import codecs
import itertools
import time
import os
import gensim
from nltk.util import ngrams


class Config(object):
    data_path = './dataset'
    train_file = 'cleaned_tweets_train'
    dev_file = 'cleaned_tweets_valid'
    test_file = 'cleaned_tweets_test'

    x_train_np_file = 'x_train.npy'
    y_train_np_file = 'y_train.npy'
    x_valid_np_file = 'x_valid.npy'
    y_valid_np_file = 'y_valid.npy'
    x_test_np_file = 'x_test.npy'
    y_test_np_file = 'y_test.npy'
    embedding_matrix_np_file = 'embedding_matrix.npy'
    word2id_np_file = 'word2id.npy'

    embedding_file = './tweets_sg_300/tweets_sg_300'
    embedding_size = 300
    n_words_per_tweet = 80
    pos_class_freq = 0.24
    neg_class_freq = 0.51
    neutral_class_freq = 0.25


class TweetsParser(object):

    def __init__(self, parse_from_scratch=True):
        self.config = Config()

        print "Dataset initializing..."
        start = time.time()
        self.class_label_to_int = {'POS': 0, 'NEG': 1, 'NEUTRAL': 2}

        if parse_from_scratch:
            self._parse_from_scratch()
            self._save_data()

        else:
            self._load_from_files()

        print "Parsing Done.\n took {:.2f} seconds\n".format(time.time() - start)

    def _parse_from_scratch(self):
        """
        Parses the dataset files and prepares the training, validation, testing sets
        in addition to the word embeddings.
        """
        print "Loading word embeddings model..."
        self.embeddings_model = gensim.models.Word2Vec.load(self.config.embedding_file)
        self.generated_embeddings = {}
        self.word2id = {}

        print "\nParsing Training data...\n"
        x_train, y_train = self._get_x_y(os.path.join(self.config.data_path, self.config.train_file))
        print "\nParsing Validation data...\n"
        x_valid, y_valid = self._get_x_y(os.path.join(self.config.data_path, self.config.dev_file))
        print "\nParing testing data...\n"
        x_test, y_test = self._get_x_y(os.path.join(self.config.data_path, self.config.test_file))

        # word2id should have been populated while reading dataset
        self.vocab_size = len(self.word2id)

        print "\nPreparing word embeddings for dataset...\n"
        self.embeddings_matrix = self._prepare_word_embeddings()

        # add zeros at the end of the embeddings matrix to be used in padding
        zeros_padding = np.zeros((1, self.config.embedding_size))
        self.embeddings_matrix = np.concatenate((self.embeddings_matrix, zeros_padding), axis=0)
        self.vocab_size += 1

        self.padding_index = len(self.embeddings_matrix) - 1

        # add index of zero-padding to the end of each tweet (to be zero-padded in the embedding layer)
        self.x_train = np.array(self._add_padding(x_train))
        self.x_valid = np.array(self._add_padding(x_valid))
        self.x_test = np.array(self._add_padding(x_test))

        self.y_train = np.array(y_train)
        self.y_valid = np.array(y_valid)
        self.y_test = np.array(y_test)

    def _load_from_files(self):
        self._init_data()
        self.vocab_size = len(self.embeddings_matrix)

    def _get_x_y(self, filename):
        """
        Takes file name, reads the tweets, convert each tweet to set of indices,
        each index represents the index of the word in the vocab
        :param filename: tweets file name
        :return:
            x: array of indices (first dim corresponds to tweets, second dim corresponding to indices representing tweet words).
            y: class label corresponding to each tweet.
        """

        x_tweets = []
        y = []

        tweets_file = codecs.open(filename, "r", "utf-8")
        for line in tweets_file:
            tokens = line.strip().split("\t")
            x_tweets.append(tokens[0])
            y.append(self.class_label_to_int[tokens[1]])

        x = self._vectorize(x_tweets)
        return x, y

    def _vectorize(self, examples):
        """
        Takes a list of sentences(tweets), and returns for each sentence a list of indices, corresponding to
        the index of each word in the vocab. Builds the word2int dict while vectorizing tweets.
        :param examples: list of sentences
        :return: 2D array of indices
        """
        vectorized_examples = []
        for example in examples:
            word_indices = []
            for word in example.split():
                word = word.strip()
                if word in self.word2id:
                    word_indices.append(self.word2id[word])
                else:
                    # check if it has an embedding
                    if word in self.embeddings_model.wv.vocab:
                        # append word to word2id, and append word index to the sentence
                        self._append_word_and_update_word2id(word, word_indices)

                    elif word in self.generated_embeddings:
                        # append word to word2id, and append word index to the sentence
                        self._append_word_and_update_word2id(word, word_indices)

                    else:
                        # remove repetitions in the word and check
                        word_no_rep = ''.join(ch for ch, _ in itertools.groupby(word))
                        if word_no_rep in self.embeddings_model.wv.vocab:
                            # append word to word2id, and append word index to te sentence
                            self._append_word_and_update_word2id(word, word_indices)
                        else:
                            # try to generate embedding for the word
                            generated_word_embedding = self._generate_embedding_for_word(word)
                            if generated_word_embedding is not None:
                                # append word
                                self._append_word_and_update_word2id(word, word_indices)

                                # append embedding
                                self.generated_embeddings[word] = generated_word_embedding

                            # else discard this word
            # we may find other ways to clean/tokenize the word,
            # may we found an embedding before generating one

            vectorized_examples.append(word_indices)

        return vectorized_examples

    def _append_word_and_update_word2id(self, word, word_indices):

        """
        Utility method for adding a new word to a list and updating word2id dict.
        """
        word_indices.append(len(self.word2id))
        self.word2id[word] = len(self.word2id)

    def _add_padding(self, x):
        """
        :param x: the array that needs to be padded.
        :return: x concatenated with index of the padding word (word with all zeros embedding).
        """
        padded_x = []
        for instance in x:
            needed_padding_len = self.config.n_words_per_tweet - len(instance)
            if needed_padding_len < 0:
                print "Tweet exceeded %d words, it has length of: %d" % (self.config.n_words_per_tweet, len(instance))
                padded_x.append(instance[:needed_padding_len])
            else:
                padding_indices = np.empty(needed_padding_len, dtype=int)
                padding_indices.fill(self.padding_index)
                padded_x.append(np.concatenate((instance, padding_indices)))
        return padded_x

    def _prepare_word_embeddings(self):
        """
        :return: embeddings_matrix: 2D matrix representing word embeddings for each word in the vocab
        (difference between it, and the embeddings_model, is that it is an array not dict, so as to be suitable
        for the keras model).
        It considers both pre-trained embeddings and generated word embeddings.
        """
        embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (self.vocab_size, self.config.embedding_size)),
                                       dtype='float32')
        for word in self.word2id:
            i = self.word2id[word]
            if word in self.embeddings_model.wv.vocab:
                embeddings_matrix[i] = self.embeddings_model.wv[word]
            elif word in self.generated_embeddings:
                embeddings_matrix[i] = self.generated_embeddings[word]
            else:
                word_no_rep = ''.join(ch for ch, _ in itertools.groupby(word))
                if word_no_rep in self.embeddings_model.wv.vocab:
                    embeddings_matrix[i] = self.embeddings_model.wv[word_no_rep]
        return embeddings_matrix

    def _generate_embedding_for_word(self, word):
        """
        :param word: the unknown word to generate embedding for.
        :return: the average of the found embeddings for each of the n-grams of the word if any, or None otherwise.
        """
        characters = [c for c in word]
        min_n = 3
        num_found_embeddings = 0
        generated_embedding = np.zeros(self.config.embedding_size)
        for n in range(min_n, len(word)):
            if num_found_embeddings:
                break
            all_ngrams = ngrams(characters, n)
            for element in all_ngrams:
                # convert to string
                token = ''.join(element)
                if (token in self.embeddings_model.wv.vocab):
                    # print "embedding found for sub-token: ", token
                    num_found_embeddings += 1
                    generated_embedding += self.embeddings_model.wv[token]
        if not num_found_embeddings:
            return None
        print "generated embedding for word: ", word
        return generated_embedding / np.array([float(num_found_embeddings)])

    def _save_data(self):
        """
        Saves x, y for training, validation and testing sets into numpy file.
        Also saves the embedding matrix and word2id map.
        :return:
        """
        # save x and y for training data
        self._save_data_np_to_file(self.x_train, self.config.data_path, self.config.x_train_np_file)
        self._save_data_np_to_file(self.y_train, self.config.data_path, self.config.y_train_np_file)
        # save x and y for validation data
        self._save_data_np_to_file(self.x_valid, self.config.data_path, self.config.x_valid_np_file)
        self._save_data_np_to_file(self.y_valid, self.config.data_path, self.config.y_valid_np_file)
        # save x and y for testing data
        self._save_data_np_to_file(self.x_test, self.config.data_path, self.config.x_test_np_file)
        self._save_data_np_to_file(self.y_test, self.config.data_path, self.config.y_test_np_file)

        # save word2id map
        self._save_data_np_to_file(self.word2id, self.config.data_path, self.config.word2id_np_file)

        # save embeddings matrix
        self._save_data_np_to_file(self.embeddings_matrix, self.config.data_path, self.config.embedding_matrix_np_file)

    def _save_data_np_to_file(self, array, directory, filename):
        out_file = codecs.open(os.path.join(directory, filename), "w")
        np.save(out_file, array)
        out_file.close()

    def _init_data(self):
        """
        Reads and initializes:
         - x and y for training, validation and testing.
         - embedding matrix
         - word2id dictionary

        :return:
        """
        self.x_train = self._read_np_from_file(self.config.data_path, self.config.x_train_np_file)
        self.y_train = self._read_np_from_file(self.config.data_path, self.config.y_train_np_file)

        self.x_valid = self._read_np_from_file(self.config.data_path, self.config.x_valid_np_file)
        self.y_valid = self._read_np_from_file(self.config.data_path, self.config.y_valid_np_file)

        self.x_test = self._read_np_from_file(self.config.data_path, self.config.x_test_np_file)
        self.y_test = self._read_np_from_file(self.config.data_path, self.config.y_test_np_file)

        self.word2id = self._read_np_from_file(self.config.data_path, self.config.word2id_np_file).item()

        self.embeddings_matrix = self._read_np_from_file(self.config.data_path, self.config.embedding_matrix_np_file)

    def _read_np_from_file(self, directory, filename):
        in_file = codecs.open(os.path.join(directory, filename), "r")
        data = np.load(in_file)
        in_file.close()
        return data
