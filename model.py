import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from preprocessing import clean_dataframe

train_data = clean_dataframe(pd.read_csv("data\\train.csv"))
validation_data = clean_dataframe(pd.read_csv("data\\validation.csv"))
test_data = clean_dataframe(pd.read_csv("data\\test.csv"))

# show_dialogue_summary_word_count(train_data)
# show_dialogue_summary_word_count(validation_data)
# show_dialogue_summary_word_count(test_data)

max_len_dialogue = 150
max_len_summary = 50

# prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(train_data["cleaned_dialogue"]))

# convert text sequences into integer sequences
x_tr = x_tokenizer.texts_to_sequences(train_data["cleaned_dialogue"])
x_val = x_tokenizer.texts_to_sequences(validation_data["cleaned_dialogue"])

# padding zero upto maximum length
x_tr = pad_sequences(x_tr, maxlen=max_len_dialogue, padding="post")
x_val = pad_sequences(x_val, maxlen=max_len_dialogue, padding="post")

x_voc_size = len(x_tokenizer.word_index) + 1
print(x_voc_size)
# -----------------------------------------------------------------------
# preparing a tokenizer for summary on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(train_data["cleaned_summary"]))

# convert summary sequences into integer sequences
y_tr = y_tokenizer.texts_to_sequences(train_data["cleaned_summary"])
y_val = y_tokenizer.texts_to_sequences(validation_data["cleaned_summary"])

# padding zero upto maximum length
y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding="post")
y_val = pad_sequences(y_val, maxlen=max_len_summary, padding="post")

y_voc_size = len(y_tokenizer.word_index) + 1
print(y_voc_size)
