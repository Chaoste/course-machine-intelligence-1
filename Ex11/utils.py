import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import os
from scipy import ndimage
from image_transformations import clipped_zoom, rotate, noise_filter, gaussian_filter, elastic_transform

def read(path):
    X, y = sklearn.datasets.load_svmlight_file(path, 784, dtype=np.float32)
    return dict(images=np.array(X.todense()).reshape((500, 784)), labels=y)

def create_validation_set(data):
    # Split up training set to have a "testing" set with known labels
    data['test']['images'] = data['train']['images'][-100:]
    data['test']['labels'] = data['train']['labels'][-100:]
    data['train']['images'] = data['train']['images'][:-100]
    data['train']['labels'] = data['train']['labels'][:-100]
    return data

def add_morphed_data(data, zoom_range=np.arange(1.1, 1.31, 0.1),
        angle_range=range(-20, 20, 5), noise_range=np.arange(0, 0.21, 0.1),
        sigma_range=np.arange(0, 1.1, 0.3),
        elastic_alpha_range=range(5, 13, 2),
        elastic_sigma_range=np.arange(2, 2.51, 0.2)):
    # Create synthetic test data by applying zoom and rotation
    new_data = []
    for zoom_level in zoom_range:
        for angle in angle_range:
            for noise_level in noise_range:
                for sigma in sigma_range:
                    for ealpha in elastic_alpha_range:
                        for esigma in elastic_sigma_range:
                            zooming = (clipped_zoom(x, zoom_level) for x in data['train']['images'])
                            rotating = (rotate(x, angle) for x in zooming)
                            noise = (noise_filter(x, noise_level) for x in rotating)
                            gaussian = (gaussian_filter(x, sigma) for x in noise)
                            elastic = (elastic_transform(x, ealpha, esigma, reshape=True) for x in gaussian)
                            new_data.append(list(elastic))
    print('{} mutations for each image'.format(len(new_data)))
    data['train']['images'] = np.concatenate(new_data)
    data['train']['labels'] = np.array(list(data['train']['labels'].data) * len(new_data))
    return data

def write(algo, labels):
    # Ensure that you're using unix newlines
    with open('./predictions_{}'.format(algo), 'w', newline='\n') as file_out:
        file_out.write(
            '\n'.join([str(int(x)) for x in labels])
        )

def plot_img(ax, img, title):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(title)

def plot_labels(ax, labels, title, color='#7070F0'):
    labels = np.array(labels, dtype=np.int)
    counts = np.bincount(labels)
    ax.bar(range(len(counts)), counts, color=color)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks(np.arange(len(counts)) + 0.4)
    ax.xaxis.set_ticklabels(np.arange(len(counts)))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.set_title(title)

def compare_results(results):
    count = len(results) + 1
    colors = ['green', 'magenta', 'orange', 'purple']
    fig, axes = plt.subplots(1, count, figsize=(13, 2))
    plot_labels(axes[0], data['train']['labels'], 'Train')
    for ax, result, color in zip(axes.flat[1:], results, colors[:count-1]):
        title, labels = result
        plot_labels(ax, labels, title)
    fig.tight_layout()

def execute_algo(algo, title, train=True, **kwargs):
    data = dict(train=read('digits/digit_train'), test=read('digits/digit_test'))
    if train:
        # Create synthetic test data by scaling and rotating
        images_rotated = [rotate(x) for x in data['train']['images']]
        images_rescaled = [clipped_zoom(x) for x in data['train']['images']]
        images_rotres = [clipped_zoom(x) for x in images_rotated]
        data['train']['images'] = np.concatenate([data['train']['images'], images_rotated, images_rescaled, images_rotres])
        data['train']['labels'] = np.array(list(data['train']['labels'].data) * 4)
        algo.fit(data['train']['images'], data['train']['labels'], **kwargs)
    predictions = algo.eval(data['test']['images'])

    # Exercise 11.1.a
    fig, axes = plt.subplots(1, 8, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        img = data['test']['images'][i].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title('{}: {:.0f}'.format(title, predictions[i]))
    fig.tight_layout()

    write('{}.txt'.format(title), predictions)
