import pandas as pd
from keras import backend as K
from keras.layers import (
    LSTM,
    Attention,
    Concatenate,
    Dense,
    Embedding,
    Input,
    TimeDistributed,
)
from keras.models import Model
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
# ------Encoder-----------------------------------------------------------------
K.clear_session()
word_vector_dim = 500

encoder_inputs = Input(shape=(max_len_dialogue,))
encoder_embedding = Embedding(x_voc_size, word_vector_dim, trainable=True)
encoder_embedding_inputs = encoder_embedding(encoder_inputs)

# LSTM 1
encoder_lstm1 = LSTM(word_vector_dim, return_sequences=True, return_state=True)
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embedding_inputs)

# LSTM 2
encoder_lstm2 = LSTM(word_vector_dim, return_sequences=True, return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# LSTM 3
encoder_lstm3 = LSTM(word_vector_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)
# ------Decoder-----------------------------------------------------------------
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(y_voc_size, word_vector_dim, trainable=True)
decoder_embedding_inputs = decoder_embedding(decoder_inputs)

# LSTM using encoder_states as initial state
decoder_lstm = LSTM(word_vector_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
    decoder_embedding_inputs, initial_state=[state_h, state_c]
)
# ------Attention Layer-----------------------------------------------------------------
attention = Attention()
attention_outputs = attention([decoder_outputs, encoder_outputs])

concatenate = Concatenate(axis=-1)
decoder_concat_input = concatenate([decoder_outputs, attention_outputs])
# ------Dense layer-----------------------------------------------------------------
decoder_dense = TimeDistributed(Dense(y_voc_size, activation="softmax"))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
