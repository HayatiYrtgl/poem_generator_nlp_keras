import spacy
import numpy as np
import string
import joblib
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint, CSVLogger


# class for nlp
class NLP:
    """This class includes preprocessing and training"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg", disable=["parse", "tag", "ner"])

        self.nlp.max_length = 6697076

        self.poems = open("../DATASET/poems/sonnet_poems.txt", "r", encoding="utf-8").read()

        # tokens
        self.tokens = self.preprocess(self.poems)

        # tokenizer
        self.tokenizer = Tokenizer()

        # sequence length
        self.seq_len = None

        # x and y
        self.X, self.y, self.vocab_size = self.sequence_data(window_size=20)

        # model
        self.model = self.create_model(vocab_size=self.vocab_size + 1, seq_len=self.X.shape[1])

        self.model.fit(self.X, self.y, epochs=200, batch_size=32, callbacks=[ModelCheckpoint(filepath="models/poem",
                                                                                            monitor="accuracy"),
                                                                            CSVLogger(filename="log_of_poem2.csv")])

    # preprocess
    def preprocess(self, document_text):
        return [token.text.lower() for token in self.nlp(document_text) if
                token.text not in string.punctuation + "\n \n\n--..."]

    # sequence data
    def sequence_data(self, window_size: int) -> np.array:
        # sequence list
        sequence = []

        # window size
        window_size = window_size

        # for loop
        for i in range(len(self.tokens) - window_size):
            selected = self.tokens[i:i + window_size]

            sequence.append(selected)

        # fitted sequence
        self.tokenizer.fit_on_texts(sequence)

        joblib.dump(self.tokenizer, "../tokenizers/tokenizer_sonnet.pkl")

        # vocab size
        vocab_size = len(self.tokenizer.word_counts)

        # transformed
        sequences = self.tokenizer.texts_to_sequences(sequence)

        # turn sequence to matrix
        matrix = np.array(sequences)

        # x, y splitting
        X, y = matrix[:, :-1], matrix[:, -1]

        # y to categorical
        y = to_categorical(y, num_classes=vocab_size + 1)

        # sequence len
        self.seq_len = window_size

        return X, y, vocab_size

    # create model
    @staticmethod
    def create_model(vocab_size, seq_len):
        # model
        model = Sequential()

        model.add(Embedding(input_dim=vocab_size + 1,
                            output_dim=seq_len,
                            input_length=seq_len))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(vocab_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        return model


c = NLP()
