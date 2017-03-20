from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.utils.visualize_util import plot

num_hidden_units_mlp = 1024
num_hidden_units_lstm = 512
img_dim = 900
word_vec_dim = 300
max_len = 100
nb_classes = 1000

model = Sequential()
model.add(Dense(num_hidden_units_mlp, input_dim=img_dim + word_vec_dim, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(num_hidden_units_mlp, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(num_hidden_units_mlp, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, init='uniform'))
model.add(Activation('softmax'))

# sherlock_model = Sequential()
# sherlock_model.add(Reshape((img_dim,), input_shape=(img_dim,)))
#
# language_model = Sequential()
# language_model.add(LSTM(output_dim=num_hidden_units_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim)))
# language_model.add(LSTM(output_dim=num_hidden_units_lstm, return_sequences=True))
# language_model.add(LSTM(output_dim=num_hidden_units_lstm, return_sequences=False))
#
# model = Sequential()
# model.add(Merge([language_model, sherlock_model],
#                 mode='concat', concat_axis=1))
# model.add(Dense(num_hidden_units_mlp, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(num_hidden_units_mlp, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.summary()
model.save('NaiveSherlock.hd5')
