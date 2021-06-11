import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np


def draw_plot(X, Y):
    c = ['red' if y < 0.5 else 'blue' for y in Y]
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=c)
    plt.show()
    plt.close(fig)


model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

X = np.random.randn(500, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0) * 1.

draw_plot(X, Y)

model.fit(X, Y, batch_size=10, epochs=10)

nX = np.random.randn(500, 2)
nY = model.predict_proba(nX)

draw_plot(nX, nY)

model.fit(X, Y, batch_size=10, epochs=100)

nX = np.random.randn(500, 2)
nY = model.predict_proba(nX)

draw_plot(nX, nY)

xY = np.logical_xor(nX[:, 0] > 0, nX[:, 1] > 0) * 1.

err = 0
for i in range(len(nY)):
    if nY[i] > 0.5 and xY[i] < 0.5:
        err += 1
    if nY[i] < 0.5 and xY[i] > 0.5:
        err += 1

print(err, len(nY), err/len(nY))
