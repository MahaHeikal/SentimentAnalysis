from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import time

from TweetsParser import TweetsParser


class LSTM_Config(object):
    """Holds model hyperparams for LSTM.
    """

    def __init__(self, lstm_hidden_size=200, dropout=0.5, lr=0.001, dense_hidden_size=30, batch_size=50, n_epochs=10):
        self.n_words_per_tweet = 80
        self.n_classes = 3
        self.n_epochs = n_epochs
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout = dropout
        self.lr = lr
        self.dense_hidden_size = dense_hidden_size
        self.batch_size = batch_size
        self.model_name = "lstm__lstm_size_{}__lr_{}__dropout_{}__dense_size_{}_{}padding" \
            .format(lstm_hidden_size, lr, dropout, dense_hidden_size, self.n_words_per_tweet)


class LSTM_Model(object):

    def __init__(self, config, parser):
        self.config = config
        self.weights_path = "model_weights/{}_padding/best-weights-{}.hdf5".format(config.n_words_per_tweet, config.model_name)
        self.parser = parser
        self.model = self._build_model()


    def _build_model(self):

        model = Sequential()
        embedding_layer = Embedding(self.parser.vocab_size, self.parser.config.embedding_size,
                                    weights=[self.parser.embeddings_matrix],
                                    input_length=self.config.n_words_per_tweet,
                                    trainable=False)
        model.add(embedding_layer)
        model.add(Dropout(self.config.dropout, seed=1))
        model.add(Bidirectional(LSTM(self.config.lstm_hidden_size), merge_mode='concat'))
        model.add(Dropout(self.config.dropout, seed=1))
        model.add(Dense(self.config.dense_hidden_size))
        model.add(Dropout(self.config.dropout, seed=1))
        model.add(Activation('relu'))
        model.add(Dense(self.config.n_classes, activation="softmax"))

        # model.summary()

        optimizer = Adam(self.config.lr)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model


    def fit_model(self):
        self.history = AccuracyHistory()
        checkpoint = ModelCheckpoint(self.weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        start = time.time()

        self.model.fit(self.parser.x_train,
                       self.parser.y_train,
                       batch_size=self.config.batch_size,
                       epochs=self.config.n_epochs,
                       verbose=1,
                       validation_data=(self.parser.x_valid, self.parser.y_valid),
                       callbacks=[self.history, checkpoint])


        print ('Max validation acc: ', self.history.max_acc)
        print ("Validation loss: ", self.history.val_loss)

        print "\nDone, total time: {:.2f} mins\n".format((time.time() - start) / 60.0)


if __name__ == "__main__":
    tweets_parser = TweetsParser(False)
    lstm_model = LSTM_Model(config, parser)
    lstm_model.fit_model()


