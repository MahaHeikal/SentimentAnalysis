# CNN and LSTM model for Arabic tweets sentiment analysis

In 2018 I published the paper ["Sentiment Analysis of Arabic Tweets using Deep Learning"](https://www.sciencedirect.com/science/article/pii/S1877050918321689), since then I wanted to refactor the code and share a clean, easy to run version of it, unfortunately I didn't have a chance to do so. But since I was asked multiple times to share the code so I decided to publish it as it is hoping that it helps or gives an insight to whoever is interesed in impelementing a similar model. So please exuecuse my Java accent in the python code :)

The main steps are:
1. The dataset is cleaned as described in the paper and split randomly into training, testing and validation sets.
2. The split sets are prepared and transformed into a format suitable for the model. The model expects the dataset to contain indices of the words not the words themselves. So, all words in the dataset are read and each word is given an index. Then a word2id map is built to represent the mapping from each word to it's given index. Also the training, validation and test sets are coverted into X and Y vectors. X contains the tweets (each line corresponds to one tweet and contains indices of the words) and Y contains the label (each line contains the label corresponding to that tweet). Moreover, the Aravec word embeddings are also read and converted into a matrix, each row represents the word embedding of one word (sorted in the same order as the indices of the words. e.g. first row in the embedding corresponds to word with given index 0). After preparing these files they are saved to the disk so that they can be loaded later without doing this step again in every run. This is done in "TweetsParser.py"
3. In order to train the model, we need:
- training, validation and test sets in the form of X, Y as described above
- the word embedding matrix
so a TweetsParser object is created which prepares these files (either doing the parsing from scratch or loading them from the saved files)
then they are passed to the model, which does the training and the fitting of the model.

