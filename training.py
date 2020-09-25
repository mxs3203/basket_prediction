from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow.keras.utils as ku 
import numpy as np
import tensorflow as tf
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)
checkpoints = tf.keras.callbacks.ModelCheckpoint("output/", monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=False, mode='auto', save_freq='epoch')


class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.savefig("loss.png")
        plt.close()


plot_losses = PlotLosses()

def tokenise_words(inputs):
    tokenizer = Tokenizer()
    sentances = []

    for file in inputs:
        with open(file) as csvfile:
            lines = csvfile.readlines()
            for l in lines:
                l = l.replace(" ","_").replace("-", "_")
                sentances.append(l.replace(",", " ").replace("&", "").strip())


    print("Example Input: ")
    print(sentances[11])


    corpus = sentances


    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print('Total unique words: ',total_words )
    # create input sequences using list of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    # create predictors and label
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

    label = ku.to_categorical(label, num_classes=total_words)
    print('Combinations: ', len(predictors))
    return predictors, label, max_sequence_len, total_words, tokenizer, sentances

def make_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dense(total_words/2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='categorical_accuracy')
    return model


def model_fit(model, predictors, label, epoch=100):
    history = model.fit(predictors, label,epochs=epoch, verbose=1, batch_size=128, callbacks=[early_stop, checkpoints, plot_losses], validation_split=0.2)
    return history

def make_plot(history):
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.savefig('final_loss.png')

def main():
    result = glob.glob('data/*.csv')
    inputs = result
    predictors, label, max_sequence_len, total_words,tokenizer, sentences = tokenise_words(inputs)
    model = make_model(total_words, max_sequence_len)
    history = model_fit(model, predictors, label, epoch =300)
    print()
    print()
    for i in sentences[1:10]:
        print("Original sentence: ", i)

        inp = " ".join(i.split(" ")[:-1])
        exp_out = "".join(i.split(" ")[-1])

        print("Input to model: ", inp)
        print("Expected output: ", exp_out)
        print("Output from Model : ", make_tekst(inp, tokenizer, max_sequence_len, model))
    make_plot(history)
    print()
    # print("Done...")

def make_tekst(seed_text, tokenizer, max_sequence_len, model):
    next_words = 2
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return (seed_text)
if __name__ == "__main__":
    main()
