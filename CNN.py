from keras.layers.embeddings import Embedding
from keras import Sequential
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Merge
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score

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
        self.weights_path = "cnn_model_best_weights.hdf5"
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
        checkpoint = ModelCheckpoint(self.weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        self.model.fit(train_set,
                       self.parser.y_train,
                       batch_size=self.config.batch_size,
                       epochs=self.config.n_epochs,
                       verbose=1,
                       validation_data=(valid_set, self.parser.y_valid),
                       callbacks=[self.history, checkpoint])


    def evaluate_model(self):
        self.model.load_weights(self.weights_path)
        test_set = [self.parser.x_test, self.parser.x_test, self.parser.x_test]
        score = self.model.evaluate(test_set, self.parser.y_test, batch_size=self.config.batch_size, verbose=1)
        print('Test Loss:', score[0])
        print('Test accuracy:', score[1])
        return score[0], score[1]


    def evaluate_with_metrics(self):
        self.model.load_weights(self.weights_path)
        test_set = [self.parser.x_test, self.parser.x_test, self.parser.x_test]
        y_pred = self.model.predict_classes(test_set)

        f1_weighted = f1_score(self.parser.y_test, y_pred, average='weighted')
        print "F1 weighted measure is {}".format(f1_weighted)

        accuracy = accuracy_score(self.parser.y_test, y_pred)
        print "Accuracy is {}".format(accuracy)

        precison = precision_score(self.parser.y_test, y_pred)
        print "Precision is {}".format(precison)

        print ("Num training examples: ", len(self.parser.x_train))
        print ("Num validation examples: ", len(self.parser.x_valid))
        print ("Num testing examples: ", len(self.parser.x_test))

        print np.bincount(self.parser.y_test)

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
    tweets_parser = TweetsParser(True)
    cnn_model = CNN_Model(tweets_parser)
    cnn_model.fit_model()
    cnn_model.evaluate_with_metrics()

