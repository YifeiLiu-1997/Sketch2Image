"""
    my_utils:
    1. some function to use in tensorflow
"""

import cv2
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from keras.preprocessing.image import img_to_array
import tensorflow as tf


def sorted_path(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def expand_image_array(path, path_list, size=(256, 256)):
    img_array = []
    for i in tqdm(path_list):
        image = cv2.imread(path + '/' + i, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        image = image.astype('float32') / 255.0
        img_array.append(img_to_array(image))

        # horizontal flip
        img1 = cv2.flip(image, 1)
        img_array.append(img_to_array(img1))
        # vertical flip
        img2 = cv2.flip(image, -1)
        img_array.append(img_to_array(img2))
        # vertical flip
        img3 = cv2.flip(image, -1)
        # horizontal flip
        img3 = cv2.flip(img3, 1)
        img_array.append(img_to_array(img3))
        # rotate clockwise
        img4 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        img_array.append(img_to_array(img4))
        # flip rotated image
        img5 = cv2.flip(img4, 1)
        img_array.append(img_to_array(img5))
        # rotate anti clockwise
        img6 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_array.append(img_to_array(img6))
        # flip rotated image
        img7 = cv2.flip(img6, 1)
        img_array.append(img_to_array(img7))

    return img_array


# show example
def plot_images(image, sketches):
    plt.figure(figsize=(7, 7))
    plt.subplot(1, 2, 1)
    plt.title('Sketches ', color='green', fontsize=20)
    plt.imshow(sketches)
    plt.subplot(1, 2, 2)
    plt.title('Image', color='black', fontsize=20)
    plt.imshow(image)

    plt.show()


def show_images(real, sketch, predicted):
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    plt.title("Image", fontsize=15, color='Lime')
    plt.imshow(real)
    plt.subplot(1, 3, 2)
    plt.title("sketch", fontsize=15, color='Blue')
    plt.imshow(sketch)
    plt.subplot(1, 3, 3)
    plt.title("Predicted", fontsize=15, color='black')
    plt.imshow(predicted)

    plt.show()


def show_my_images(img_file, model, size=256):
    img_show = cv2.imread(img_file)
    img_show = cv2.resize(img_show, (size, size))
    img_show = img_show.astype('float32') / 255.0
    img_show = img_to_array(img_show)

    img_pred = cv2.imread(img_file)
    img_pred = cv2.resize(img_pred, (size, size))
    img_pred = img_pred.astype('float32')
    img_pred = img_to_array(img_pred)
    img_pred[img_pred > 80] = 255
    img_pred = img_pred / 255.0

    predict = model.predict(img_pred.reshape(1, size, size, 3)).reshape((size, size, 3))
    print('my pred', predict.shape)
    plt.subplot(1, 2, 1)
    plt.title('My sketch')
    plt.imshow(img_show)
    plt.subplot(1, 2, 2)
    plt.title('Predict')
    plt.imshow(predict)

    plt.show()


def plot_metrics(history, metric, show=False):
    """
    :param history: tensorflow model
    :param metric: tensorflow model metric -> 'loss', 'accuracy'
    :return: nothing
    """
    if show:
        plt.figure(figsize=(20, 8))
        train_metric = history.history[metric]
        val_metric = history.history['val_' + metric]
        plt.plot(range(1, int(len(history.history[metric])+1)), train_metric, 'bo-', label='Training %s' % metric)
        plt.plot(range(1, int(len(history.history[metric])+1)), val_metric, 'ro--', label='Validation %s' % val_metric)
        plt.title("Training and Validation " + metric)
        plt.xlabel('Epochs')
        plt.xticks(range(1, int(len(history.history[metric])+1)))
        plt.ylabel(metric)
        plt.legend(['train ' + metric, 'val ' + metric])
        plt.show()
    else:
        return


def save_all_model(model, save_path, save=False):
    if save:
        model.save(save_path)


def load_all_model(model, save_path, load=False):
    if load:
        fresh_model = tf.keras.models.load_model(save_path)
        return fresh_model
    return model