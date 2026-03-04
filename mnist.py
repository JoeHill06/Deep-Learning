# currently scoring I:290 Train-Err:0.383 Train-Acc:0.851 Test-Err:0.428 Test-Acc:0.797%  

import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000]) 

one_hot_labels = np.zeros((len(labels),10))

for i,l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

test_images = x_test.reshape(len(x_test), 28*28) /255
test_labels = np.zeros((len(y_test), 10))
for i,l in enumerate(y_test):
    test_labels[i][l] = 1

np.random.seed(1)
relu = lambda x: (x>=0) * x
relu2deriv = lambda x: x>=0
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)
def tanh(x):
    return np.tanh(x)
def tanh_deriv(output):
    return 1 - output ** 2   # output is already tanh(x)

hidden_size, pixels_per_image, num_labels = (100,784,10)
alpha, iterations = (2, 300)
batch_size = 100

weights_0_1 = 0.02*np.random.random((pixels_per_image, hidden_size)) - 0.01
weights_1_2 = 0.2*np.random.random((hidden_size, num_labels)) -0.1

for j in range(iterations):
    error, correct_cnt = (0.0,0)

    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i*batch_size), ((i+1)*batch_size))

        layer_0 = images[batch_start:batch_end]
        layer_1 = tanh(np.dot(layer_0, weights_0_1))
        dropout_mask = np.random.randint(2,size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = softmax(np.dot(layer_1,weights_1_2))

        error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1])) == np.argmax(labels[batch_start+k:batch_start+k+1])

        layer_2_delta = (labels[batch_start:batch_end] - layer_2)/(batch_size * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh_deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
    
    if (j%10 == 0):
        test_error = 0.0
        test_correct_cnt = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            layer_2 = softmax(np.dot(layer_1, weights_1_2))

            test_error += np.sum((test_labels[i:i+1] - layer_2) **2)
            test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))

    
        sys.stdout.write("\rI:" + str(j) +
            " Train-Err:" + str(error / float(len(images)))[0:5] +
            " Train-Acc:" + str(correct_cnt / float(len(images)))[0:5] +
            " Test-Err:" + str(test_error / float(len(test_images)))[0:5] +
            " Test-Acc:" + str(test_correct_cnt / float(len(test_images)))[0:5])
        sys.stdout.flush()