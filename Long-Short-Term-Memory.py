'''

Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from Common import Common
from sklearn.neighbors import KNeighborsClassifier
from keras.optimizers import RMSprop

from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score

max_features = 20000
maxlen = 100
batch_size = 32

print('Loading data...')
X, y, video_frame_windows = Common().generate_ucf_50_dataset('frames')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb_classes = np.max(y) + 1

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen, dtype="float64")
X_test = sequence.pad_sequences(X_test, maxlen=maxlen, dtype="float64")


y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Dropout(0.2))
model.add(LSTM(128))  # try using a GRU instead, for fun
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))
#rmsprop = RMSprop(lr=1e-6)

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam')


print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
          validation_data=(X_test, y_test), show_accuracy=True)

#score, acc = model.evaluate(X_test, y_test,
#                            batch_size=batch_size,
#                            show_accuracy=True)

X = sequence.pad_sequences(X,maxlen=maxlen, dtype="float64")
pred = model.predict(X)

preds = []
y_test_result = []
acc = 0

# pegar a media dos frames preditos como a predicao para o video
for i in range(len(video_frame_windows)):
    y_test_result.append(y[acc + video_frame_windows[i]-1])
    preds.append(np.mean(y[acc: acc + video_frame_windows[i]-1]))

    acc += video_frame_windows[i]

print("accuracy : ", accuracy_score(y_test_result, preds))
print("precision : ", precision_score(y_test_result, preds,average="macro"))
