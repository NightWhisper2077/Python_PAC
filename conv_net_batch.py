import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import softmax

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Testing images shape:", test_images.shape)
print("Testing labels shape:", test_labels.shape)


def plot_images(images, labels, num_images):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()


# Визуализируем первые 25 изображений и их метки
# plot_images(train_images, train_labels, 25)


def set_padding(X, padding):
    in_shape = X.shape
    out_shape = [X.shape[0], X.shape[1], X.shape[2] + 2 * padding, X.shape[3] + 2 * padding]
    out = np.zeros(out_shape)
    for batch in range(out_shape[0]):
        for i in range(out_shape[1]):
            for j in range(in_shape[2]):
                for k in range(in_shape[3]):
                    out[batch, i, j + padding, k + padding] = X[batch, i, j, k]

    return out


def set_stride(X, stride):
    in_shape = X.shape
    out_shape = [X.shape[0], X.shape[1], 2 * X.shape[2] - 1, 2 * X.shape[3] - 1]

    out = np.zeros(out_shape)

    for batch in range(out_shape[0]):
        for i in range(out_shape[1]):
            for j in range(in_shape[2]):
                for k in range(in_shape[3]):
                    out[batch, i, j * stride, k * stride] = X[batch, i, j, k]

    return out


def set_over(x, y, kernel_size):
    out_shape = [y.shape[0], y.shape[1], y.shape[2] * kernel_size, y.shape[3] * kernel_size]
    out = np.zeros(out_shape)

    out = out * x

    return out


def CELoss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = 0

    for i in range(y_true.shape[0]):
        loss += -np.mean(y_true[i] * np.log(y_pred[i]))

    loss = loss / y_true.shape[0]

    return loss


class MySoftmax:
    def __init__(self):
        self.softmax = softmax

    def forward(self, x):
        res = []
        for i in range(x.shape[0]):
            res.append(self.softmax(x[i]))

        res = np.array(res)

        return res


class MyConv2D:
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, grad_fn=True):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.gradient = np.zeros(shape=(out_ch, in_ch, kernel_size, kernel_size))
        self.conv = np.random.normal(0, 1, size=(out_ch, in_ch, kernel_size, kernel_size))
        self.x = None

    def forward(self, X):
        self.x = X.copy()
        X = set_padding(X, self.padding)

        in_shape = X.shape

        out_shape = [in_shape[0], self.out_ch, (in_shape[2] - self.kernel_size) // self.stride + 1,
                     (in_shape[3] - self.kernel_size) // self.stride + 1]

        out = np.zeros(out_shape)
        for batch in range(in_shape[0]):
            for i in range(self.out_ch):
                for j in range(self.in_ch):
                    for h1 in range(out_shape[2]):
                        for h2 in range(out_shape[3]):
                            for k1 in range(self.kernel_size):
                                for k2 in range(self.kernel_size):
                                    out[batch, i, h1, h2] += X[batch, j, h1 * self.stride + k1, h2 * self.stride + k2] * \
                                                             self.conv[
                                                                 i, j, k1, k2]

        return out

    def backward(self, loss, lr):
        d_w = np.zeros_like(self.conv)

        out_shape = d_w.shape
        for batch in range(self.x.shape[0]):
            for i in range(out_shape[0]):
                for j in range(out_shape[1]):
                    for h1 in range(out_shape[2]):
                        for h2 in range(out_shape[3]):
                            for k1 in range(loss.shape[2]):
                                for k2 in range(loss.shape[3]):
                                    d_w[i, j, h1, h2] += self.x[batch, j, h1 + k1, h2 + k2] * loss[batch, i, k1, k2]

        d_w = d_w / self.x.shape[0]

        d_x = np.zeros_like(self.x)

        input = set_stride(loss, self.stride)

        input = set_padding(input, self.conv.shape[3] - 1)

        rot_kernel = np.rot90(self.conv, k=2)

        for batch in range(d_x.shape[0]):
            for i in range(d_x.shape[1]):
                for j in range(input.shape[1]):
                    for h1 in range(d_x.shape[2]):
                        for h2 in range(d_x.shape[3]):
                            for k1 in range(rot_kernel.shape[1]):
                                for k2 in range(rot_kernel.shape[2]):
                                    d_x[batch, i, h1, h2] += input[batch, j, h1 + k1, h2 + k2] * rot_kernel[
                                        j, i, k1, k2]

        self.conv = self.conv - lr * d_w

        return d_x


class MyReLU:
    def __init__(self, grad_fn=True):
        self.grad_fn = grad_fn
        self.gradient = None

    def forward(self, X):
        out = X

        out[out < 0] = 0

        if self.grad_fn:
            self.gradient = out.copy()
            self.gradient[self.gradient < 0] = 0
            self.gradient[self.gradient >= 0] = 1

        return out

    def backward(self, loss):
        return self.gradient * loss


class MyMaxPooling:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.gradient = None

    def forward(self, X):
        x = set_padding(X, self.padding)

        self.gradient = np.zeros_like(X)

        in_shape = x.shape

        out_shape = [in_shape[0], in_shape[1], (in_shape[2] - self.kernel_size) // self.stride + 1,
                     (in_shape[3] - self.kernel_size) // self.stride + 1]

        out = np.zeros(out_shape)
        for batch in range(in_shape[0]):
            for i in range(out_shape[1]):
                for h1 in range(out_shape[2]):
                    for h2 in range(out_shape[3]):
                        max = np.zeros(self.kernel_size * self.kernel_size)
                        for k1 in range(self.kernel_size):
                            for k2 in range(self.kernel_size):
                                max[k1 * self.kernel_size + k2] = x[
                                    batch, i, h1 * self.stride + k1, h2 * self.stride + k2]

                        out[batch, i, h1, h2] = np.max(max)
                        max_ind = np.argmax(max)

                        self.gradient[batch, i, max_ind // self.kernel_size, max_ind % self.kernel_size] = max_ind

        return out

    def backward(self, loss):
        out = set_over(self.gradient, loss, self.kernel_size)
        return out


class MyFC:
    def __init__(self, in_ch, out_ch):
        self.weights = np.random.normal(0, 1, size=[in_ch, out_ch])
        self.gradient = None
        self.x = None

    def forward(self, X):
        self.x = X.copy()
        return np.dot(X, self.weights)

    def backward(self, loss, lr):
        d_w = np.dot(self.x.T, loss)

        d_w = d_w / loss.shape[0]

        d_x = np.dot(loss, self.weights.T)

        self.weights = self.weights - lr * d_w

        return d_x


class MyFlatten:
    def __init__(self):
        self.x = None

    def forward(self, X):
        self.x = X.copy()

        out = X.reshape([X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]])

        return out

    def backward(self, loss):
        out = loss.reshape(self.x.shape)

        return out


class ModelNet:
    def __init__(self, in_ch, out_ch):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = MyConv2D(in_ch, 2, kernel_size=5, stride=1)
        self.max_pool1 = MyMaxPooling(2, stride=2)
        self.conv2 = MyConv2D(2, 4, kernel_size=3, stride=1)
        self.max_pool2 = MyMaxPooling(kernel_size=2, stride=2)
        self.flatten = MyFlatten()
        self.relu1 = MyReLU()
        self.relu2 = MyReLU()
        self.fc = MyFC(100, 10)
        self.softmax = MySoftmax()

    def forward(self, X):
        h = self.conv1.forward(X)
        # print(f'1 {h}')
        h = self.relu1.forward(h)
        # print(f'2 {h}')
        h = self.max_pool1.forward(h)
        # print(f'3 {h}')
        h = self.conv2.forward(h)
        # print(f'4 {h}')
        h = self.relu2.forward(h)
        # print(f'5 {h}')
        h = self.max_pool2.forward(h)
        # print(f'6 {h}')
        h = self.flatten.forward(h)
        # print(f'7 {h}')
        h = self.fc.forward(h)
        # print(f'8 {h}')
        h = self.softmax.forward(h)
        # print(f'9 {h}')

        return h

    def backward(self, y_pred, y_true, lr):
        loss = y_pred - y_true
        # print(loss)
        d_fc = self.fc.backward(loss, lr)
        d_flatten = self.flatten.backward(d_fc)
        d_maxPool2 = self.max_pool2.backward(d_flatten)
        d_relu2 = self.relu2.backward(d_maxPool2)
        d_conv2 = self.conv2.backward(d_relu2, lr)
        d_maxPool1 = self.max_pool1.backward(d_conv2)
        d_relu1 = self.relu1.backward(d_maxPool1)
        d_conv1 = self.conv1.backward(d_relu1, lr)


def ind2arr(i):
    a = np.zeros(10)
    a[i] = 1

    return a


def ind4arr(x):
    index = 0
    value = 0

    for i in range(len(x)):
        if x[i] > value:
            index = i
            value = x[i]

    return index


def ind4arr_batch(x):
    index = []

    for i in range(len(x)):
        index.append(ind4arr(x[i]))

    return index


def cnt_eq(x, y):
    count = 0

    for i in range(len(x)):
        if x[i] == y[i]:
            count += 1

    return count


def train(x_train, y_train, x_valid, y_valid, model, epochs, lr, count, valid_len):
    for epoch in range(epochs):
        print(f'{epoch}:')

        count_equal = 0
        running_loss = 0

        train_count = len(x_train)
        valid_count = len(x_valid)

        for i in range(train_count):
            res = model.forward(x_train[i])
            true = y_train[i]

            loss = CELoss(res, true)

            running_loss += loss

            model.backward(res, true, lr)

            res = ind4arr_batch(res)
            true = ind4arr_batch(true)

            count_equal += cnt_eq(res, true)

        print(f'My train score: {count_equal / count}. Train loss: {running_loss / train_count}')

        running_loss = 0
        count_equal = 0

        for i in range(valid_count):
            res = model.forward(x_valid[i])
            true = y_valid[i]

            loss = CELoss(res, true)

            running_loss += loss

            res = ind4arr_batch(res)
            true = ind4arr_batch(true)

            print(res, true)

            count_equal += cnt_eq(res, true)

        print(f'My valid score: {count_equal / valid_len}. Valid loss: {running_loss / valid_count}')


def split_dataset(x, y, valid_size, all_cnt):
    x_train = x[valid_size:all_cnt]
    y_train = y[valid_size:all_cnt]

    x_valid = x[0:valid_size]
    y_valid = y[0:valid_size]

    return x_train, x_valid, y_train, y_valid


def create_batch(x, y, batch_size):
    x_batch = []
    y_batch = []
    input = []
    output = []

    count = len(x)

    for i in range(count):
        input.append(np.array([x[i]]))
        output.append(ind2arr(y[i]))

        if i % batch_size == batch_size - 1:
            x_batch.append(np.array(input))
            y_batch.append(np.array(output))

            input.clear()
            output.clear()

    return x_batch, y_batch


# константы для обучения
epochs = 1000
lr = 0.01
count = 128
batch_size = 8
batch_count = count // batch_size
valid_count = 8

# препроцессинг
x_train, x_valid, y_train, y_valid = split_dataset(train_images, train_labels, valid_count, count)

x_train, y_train = create_batch(x_train, y_train, batch_size)
x_valid, y_valid = create_batch(x_valid, y_valid, batch_size)
# загрузка и создание модели
net = ModelNet(1, 10)

# обучение
train(x_train, y_train, x_valid, y_valid, net, epochs, lr, count, valid_count)
