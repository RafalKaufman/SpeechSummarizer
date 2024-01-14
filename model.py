import pandas as pd

from preprocessing import clean_dataset, show_word_count

train_data = clean_dataset(pd.read_csv("data\\train.csv"))
validation_data = clean_dataset(pd.read_csv("data\\validation.csv"))
test_data = clean_dataset(pd.read_csv("data\\test.csv"))

show_word_count(train_data)
show_word_count(validation_data)
show_word_count(test_data)

max_len_dialogue = 240
max_len_summary = 90
