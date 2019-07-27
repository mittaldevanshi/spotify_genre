import numpy as np
# import sys
# np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
# from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn import preprocessing
from sklearn.metrics import classification_report

labels = ['Pop_Rock', 'Rap', 'Blues', 'RnB', 'Folk', 'Country', 'Jazz',
          'New Age', 'Latin', 'Electronic', 'International', 'Vocal', 'Reggae']

# train_x = []
# train_y = []
# test_x = []
# test_y = []
#
# import os.path
# with open("./data/data_80_train.cls", 'r') as f:
#     lines = f.readlines()
#     n = len(lines)
#     i = 0
#     for li in lines:
#         name = li.split('\t')[0]
#         fname = "/mnt/tmp/{}.npz".format(name)
#         print(i/n * 100)
#         i += 1
#         if not os.path.isfile(fname):
#             continue
#         data = np.load(fname)
#         train_x.append(data['arr_0'])
#         train_y.append(data['arr_1'])
#
# with open("./data/data_80_test.cls", 'r') as f:
#     lines = f.readlines()
#     n = len(lines)
#     i = 0
#     for li in lines:
#         name = li.split('\t')[0]
#         fname = "/mnt/tmp/{}.npz".format(name)
#         print(i/n * 100)
#         i += 1
#         if not os.path.isfile(fname):
#             continue
#         data = np.load(fname)
#         test_x.append(data['arr_0'])
#         test_y.append(data['arr_1'])
#
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# test_x = np.array(test_x)
# test_y = np.array(test_y)
#
# np.savez("./data80unb.npz", train_x, train_y, test_x, test_y)
data = np.load("./data80unb.npz")

# data = np.load("./data2000.npz")
train_x = data['arr_0']
train_y = data['arr_1']

test_x = data['arr_2']
test_y = data['arr_3']

train_x = train_x.reshape(train_x.shape[0] * 64, 27)
test_x = test_x.reshape(test_x.shape[0] * 64, 27)
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
test_x = min_max_scaler.transform(test_x)
train_x = train_x.reshape(train_x.shape[0] // 64, 64, 27, 1)
test_x = test_x.reshape(test_x.shape[0] // 64, 64, 27, 1)

# from imblearn.over_sampling import RandomOverSampler
# train_x = train_x.reshape(train_x.shape[0], 64 * 27 * 1)
# ros = RandomOverSampler(random_state=0)
# train_x, train_y = ros.fit_resample(train_x, train_y)
# train_x = train_x.reshape(train_x.shape[0], 64, 27, 1)

class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(train_y), train_y)


def DNN():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(64, 27, 1)),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(13, activation=tf.nn.softmax)
    ])


def FCN():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu, input_shape=(64, 27, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3),
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3),
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(512, kernel_size=(3, 3),
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
        # tf.keras.layers.Conv2D(1024, kernel_size=(3, 3),
        #                        padding='same',
        #                        activation=tf.nn.relu),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(13, activation=tf.nn.softmax)
    ])


def CRNN():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                               padding='same',
                               activation=tf.nn.relu, input_shape=(64, 27, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3),
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3),
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(512, kernel_size=(3, 3),
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Reshape((-1, 512)),
        tf.keras.layers.GRU(512, return_sequences=True),
        tf.keras.layers.GRU(512),
        tf.keras.layers.Dense(13, activation=tf.nn.softmax)
    ])


# model = FCN()
# model = CRNN()
model = DNN()

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5, class_weight=class_weights)
results = model.evaluate(test_x, test_y)
print(results)

test_predictions = model.predict_classes(test_x)

# print(np.histogram(test_predictions,
#                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))

# cm = confusion_matrix(test_y, test_predictions)
# print(cm)

print(classification_report(test_y, test_predictions, target_names=labels))
