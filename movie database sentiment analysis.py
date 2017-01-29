import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
#loading the dataset from imdb and setting validation, training and testing size
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY= train
testX, testY =test
#data preprocessing
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=1000, value=0.)
#converting labls to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
#building up the LSTM network as it requires a lot of backpropagation computation
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net,optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')
#training the network and also displaying tensorboard results
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set(testX, testY), show_metric=True, batch_size=32)