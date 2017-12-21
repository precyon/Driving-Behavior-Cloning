import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
inputs = Input(shape=(3,2,4))

# Define a model consisting only of the Flatten operation
prediction = Flatten()(inputs)
model = Model(input=inputs, output=prediction)

#model = Model()
#model.add(Input(shape=(3,2,4)))
#model.add(Flatten())

X = np.arange(0,24).reshape(1,3,2,4)
print('From np')
print(X)
#[[[[ 0  1  2  3]
#   [ 4  5  6  7]]
#
#  [[ 8  9 10 11]
#   [12 13 14 15]]
#
#  [[16 17 18 19]
#   [20 21 22 23]]]]
print('From Keras')
fk = model.predict(X)
#array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
#         11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
#         22.,  23.]], dtype=float32)

print(np.reshape(fk, (3,2,4)))
