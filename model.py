import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import EarlyStopping
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
from keras.regularizers import l2
from matplotlib import pyplot

from preprocessing import clean_dataframe

train_data = clean_dataframe(pd.read_csv("data\\train.csv"))
validation_data = clean_dataframe(pd.read_csv("data\\validation.csv"))
test_data = clean_dataframe(pd.read_csv("data\\test.csv"))

# show_dialogue_summary_word_count(train_data)
# show_dialogue_summary_word_count(validation_data)
# show_dialogue_summary_word_count(test_data)

# parameters chosen based of the word_count plots
max_len_dialogue = 60
max_len_summary = 10

# -----------------------------------------------------------------------
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
encoder_lstm1 = LSTM(
    word_vector_dim, return_sequences=True, return_state=True, dropout=0.3
)
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embedding_inputs)

# LSTM 2
encoder_lstm2 = LSTM(
    word_vector_dim, return_sequences=True, return_state=True, dropout=0.3
)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# LSTM 3
encoder_lstm3 = LSTM(
    word_vector_dim, return_state=True, return_sequences=True, dropout=0.3
)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# ------Decoder-----------------------------------------------------------------
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(y_voc_size, word_vector_dim, trainable=True)
decoder_embedding_inputs = decoder_embedding(decoder_inputs)

# LSTM using encoder_states as initial state
decoder_lstm = LSTM(
    word_vector_dim, return_sequences=True, return_state=True, dropout=0.3
)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
    decoder_embedding_inputs, initial_state=[state_h, state_c]
)

# ------Attention Layer-----------------------------------------------------------------
attention = Attention()
attention_outputs = attention([decoder_outputs, encoder_outputs])

concatenate = Concatenate(axis=-1)
decoder_concat_input = concatenate([decoder_outputs, attention_outputs])

# ------Dense layer-----------------------------------------------------------------
decoder_dense = TimeDistributed(
    Dense(y_voc_size, activation="softmax", kernel_regularizer=l2(0.001))
)
decoder_outputs = decoder_dense(decoder_concat_input)

# -----------------------------------------------------------------------
# defining the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
es = EarlyStopping(monitor="val_loss", mode="min", verbose=1)
history = model.fit(
    [x_train, y_train[:, :-1]],
    y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
    epochs=20,
    callbacks=[es],
    batch_size=512,
    validation_data=(
        [x_validate, y_validate[:, :-1]],
        y_validate.reshape(y_validate.shape[0], y_validate.shape[1], 1)[:, 1:],
    ),
)
pyplot.plot(history.history["loss"], label="train")
pyplot.plot(history.history["val_loss"], label="test")
pyplot.legend()
pyplot.show()

# -----------------------------------------------------------------------
# reversing the tokenizer
reverse_target_word_index = train_summary_tokenizer.index_word
reverse_source_word_index = train_dialogue_tokenizer.index_word
target_word_index = train_summary_tokenizer.word_index

# -----------------------------------------------------------------------
# encoder inference
encoder_model = Model(
    inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c]
)

# -----------------------------------------------------------------------
# decoder inference
decoder_state_input_h = Input(shape=(word_vector_dim,))
decoder_state_input_c = Input(shape=(word_vector_dim,))
decoder_hidden_state_input = Input(shape=(max_len_dialogue, word_vector_dim))

dec_emb2 = decoder_embedding(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c]
)

# -----------------------------------------------------------------------
# attention inference
attn_out_inf = attention([decoder_outputs2, decoder_hidden_state_input])

# -----------------------------------------------------------------------
# final decoder model
decoder_inf_concat = Concatenate(axis=-1, name="concat")(
    [decoder_outputs2, attn_out_inf]
)
decoder_outputs2 = decoder_dense(decoder_inf_concat)

decoder_model = Model(
    [decoder_inputs]
    + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2],
)


# -----------------------------------------------------------------------
# generating summaries for new dialogues
def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index["start"]

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        # Get output probabilities from the model
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Get the probabilities for the next word
        output_probabilities = output_tokens[0, -1, :]

        # Sample the next word index from the probability distribution
        sampled_token_index = np.random.choice(
            np.arange(len(output_probabilities)),
            p=output_probabilities,
        )

        # Get the actual word corresponding to the sampled token index
        sampled_token = reverse_target_word_index[sampled_token_index + 1]

        # Add the sampled word to the decoded sentence
        if sampled_token != "end":
            decoded_sentence += " " + sampled_token

        if sampled_token == "end" or len(decoded_sentence.split()) >= (
            max_len_summary - 1
        ):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    newString = ""
    for i in input_seq:
        if (i != 0 and i != target_word_index["start"]) and i != target_word_index[
            "end"
        ]:
            newString = newString + reverse_target_word_index[i] + " "
    return newString


def seq2text(input_seq):
    newString = ""
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + " "
    return newString


x_validate_np = np.array(x_validate)
y_validate_np = np.array(y_validate)

for i in range(len(x_validate)):
    print("Review:", seq2text(x_validate[i]))
    print("Original summary:", seq2summary(y_validate[i]))
    print(
        "Predicted summary:",
        decode_sequence(x_validate[i].reshape(1, max_len_dialogue)),
    )
    print("\n")
