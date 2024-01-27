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

# parameters chosen based of the word_count plots
max_len_dialogue = 150
max_len_summary = 50

# tokenizing cleaned training dialogues
train_dialogue_tokenizer = Tokenizer()
train_dialogue_tokenizer.fit_on_texts(train_data["cleaned_dialogue"])

# converting dialogues into integer sequences
x_train = train_dialogue_tokenizer.texts_to_sequences(train_data["cleaned_dialogue"])
x_validate = train_dialogue_tokenizer.texts_to_sequences(
    validation_data["cleaned_dialogue"]
)

# padding sequences to the same length (from 0 up to max_len_dialogue)
x_train = pad_sequences(x_train, maxlen=max_len_dialogue, padding="post")
x_validate = pad_sequences(x_validate, maxlen=max_len_dialogue, padding="post")

x_voc_size = len(train_dialogue_tokenizer.word_index) + 1
print(x_voc_size)
# -----------------------------------------------------------------------
# tokenizing cleaned training summaries
train_summary_tokenizer = Tokenizer()
train_summary_tokenizer.fit_on_texts(train_data["cleaned_summary"])

# converting summaries into integer sequences
y_train = train_summary_tokenizer.texts_to_sequences(train_data["cleaned_summary"])
y_validate = train_summary_tokenizer.texts_to_sequences(
    validation_data["cleaned_summary"]
)

# padding sequences to the same length (from 0 up to max_len_summary)
y_train = pad_sequences(y_train, maxlen=max_len_summary, padding="post")
y_validate = pad_sequences(y_validate, maxlen=max_len_summary, padding="post")

y_voc_size = len(train_summary_tokenizer.word_index) + 1
print(y_voc_size)
