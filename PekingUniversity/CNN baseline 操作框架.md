##  CNN baseline 操作框架

``` python

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, \
    BatchNormalization, Activation, MaxPool2D, Dropout, Flatten
import numpy as np
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
import matplotlib.pyplot as plt

# cpu or gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_data(path):
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)

# 1.data
(x_train, y_train), (x_test, y_test) = load_data('./cifar-10-batches-py')
x_train, x_test = x_train/255.0, x_test/255.0
# 2. model
class MyCNN(Model):

    def __init__(self):
        super(MyCNN, self).__init__()
        self.c = Conv2D(filters=6, kernel_size=(5,5), padding='same')
        self.b = BatchNormalization()
        self.a = Activation('relu')
        self.p = MaxPool2D(pool_size=(2,2), strides=2, padding='same')
        self.d = Dropout(0.2)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.c(inputs)
        x = self.b(x)
        x = self.a(x)
        x = self.p(x)
        x = self.d(x)
        x = self.flatten(x)
        x = self.dense1(x)
        outputs = self.dense2(x)
        return outputs

model = MyCNN()
# 3. model 编译
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

# 4. 训练
# 加载模型参数续训
checkpoint_save_path = "./checkpoint/cifar10.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=256, epochs=5,
                    validation_data=(x_test, y_test), validation_freq=1, callbacks=[cp_callback])

# 保存可训练参数
model.summary()
# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

# 5.画图： loss acc

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


```