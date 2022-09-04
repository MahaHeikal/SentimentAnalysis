from keras.layers.embeddings import Embedding
from keras import Sequential
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.layers import Merge
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import numpy as np

from TweetsParser import TweetsParser


class Config(object):
    """Holds model hyperparams and data information.
    """
    n_words_per_tweet = 80
    n_classes = 3
    dropout = 0.5
    n_filters = 200
    filter_sizes = [3, 4, 5]
    hidden_size = 30
    batch_size = 50
    n_epochs = 10
    lr = 0.001


class CNN_Model(object):

    def __init__(self, parser):
        self.config = Config()
        self.parser = parser
        self.model = self._build_model()

    def _build_model(self):
        # array of convolutional layers, each with different filter size
        conv_models = []
        for filter_size in self.config.filter_sizes:
            # embedding layer
            embedding_layer = Embedding(self.parser.vocab_size, self.parser.config.embedding_size,
                                        weights=[self.parser.embeddings_matrix],
                                        input_length=self.config.n_words_per_tweet,
                                        trainable=False)

            conv_layer = Conv1D(self.config.n_filters, filter_size,
                                padding='valid', activation='relu', strides=1)
            # max pooling layer
            max_pooling_layer = GlobalMaxPooling1D()

            # we may add dropout layer here
            sub_model = Sequential([embedding_layer, conv_layer, max_pooling_layer])
            conv_models.append(sub_model)
            # sub_model.summary()

        model = Sequential()

        # merge output of max pooled values obtained form different filters.
        model.add(Merge(conv_models, mode="concat"))
        model.add(Dense(self.config.hidden_size))
        model.add(Dropout(self.config.dropout, seed=1))
        model.add(Activation('relu'))

        model.add(Dense(self.config.n_classes, activation="softmax"))

        optimizer = Adam(self.config.lr)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # print model.summary()

        return model

    def fit_model(self):
        train_set = [self.parser.x_train, self.parser.x_train, self.parser.x_train]
        valid_set = [self.parser.x_valid, self.parser.x_valid, self.parser.x_valid]

        self.history = AccuracyHistory()
        filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        self.model.fit(train_set,
                       self.parser.y_train,
                       batch_size=self.config.batch_size,
                       epochs=self.config.n_epochs,
                       verbose=1,
                       validation_data=(valid_set, self.parser.y_valid),
                       callbacks=[self.history, checkpoint])


class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


if __name__ == "__main__":
    tweets_parser = TweetsParser(False)
    cnn_model = CNN_Model(tweets_parser)
    cnn_model.fit_model()

