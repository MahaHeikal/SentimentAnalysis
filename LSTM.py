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


class LSTM_Model(object):

    def __init__(self, config, parser):
        self.config = config
        self.weights_path = "lstm_model_best_weights.hdf5"
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

    def evaluate_model(self):
        self.model.load_weights(self.weights_path)
        score = self.model.evaluate(self.parser.x_test, self.parser.y_test, batch_size=self.config.batch_size, verbose=1)
        print('Test Loss:', score[0])
        print('Test accuracy:', score[1])
        return score[0], score[1]


    def evaluate_with_metrics(self):
        self.model.load_weights(self.weights_path)
        y_pred = self.model.predict_classes(self.parser.x_test)

        f1_weighted = f1_score(self.parser.y_test, y_pred, average='weighted')
        print "F1 weighted measure is {}".format(f1_weighted)

        precison = precision_score(self.parser.y_test, y_pred)
        print "Precision is {}".format(precison)

        accuracy = accuracy_score(self.parser.y_test, y_pred)
        print "Accuracy is {}".format(accuracy)

        return accuracy, f1_weighted, precison

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []
        self.max_acc = -10000
        self.val_loss = 10000

    def on_epoch_end(self, batch, logs={}):
        current_acc = logs.get('val_acc')
        current_loss = logs.get('val_loss')
        self.acc.append(current_acc)
        self.loss.append(current_loss)
        if (current_acc > self.max_acc):
            self.max_acc = current_acc
            self.val_loss = current_loss


if __name__ == "__main__":
    config = LSTM_Config()
    tweets_parser = TweetsParser(True)
    lstm_model = LSTM_Model(config, tweets_parser)
    lstm_model.fit_model()
    lstm_model.evaluate_with_metrics()


