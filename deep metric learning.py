import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from itertools import permutations
from Plots import plot_distribution, plot_distance_map, dimension_analysis, plot_damage_locations, training_process
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = " "


def kernel_regression(x1, y1, x2, sigma):
    dist = pairwise_distances(x2, x1, squared=True)
    if HYBRID:
        softmax = np.exp(- dist[:, :-exp1_size] / sigma -
                         logsumexp(- dist[:, :-exp1_size] / sigma, axis=1)[:, np.newaxis])
        y2_ = softmax.dot(y1[:-exp1_size])
    else:
        softmax = np.exp(- dist / sigma - logsumexp(- dist / sigma, axis=1)[:, np.newaxis])
        y2_ = softmax.dot(y1)
    return y2_


def kernel_learn(x, y):
    dist = pairwise_distances(x, squared=True)
    np.fill_diagonal(dist, np.inf)
    sigma = np.logspace(-5, -1, 5)
    cost = []
    for k in range(len(sigma)):
        if HYBRID:
            softmax = np.exp(- dist[:, :-exp1_size] / sigma[k] -
                             logsumexp(- dist[:, :-exp1_size] / sigma[k], axis=1)[:, np.newaxis])
            yhat = softmax.dot(y[:-exp1_size])
        else:
            softmax = np.exp(- dist / sigma[k] - logsumexp(- dist / sigma[k], axis=1)[:, np.newaxis])
            yhat = softmax.dot(y)
        ydiff = yhat - y
        cost.append(np.mean(np.linalg.norm(ydiff, axis=1)))
    index = np.argmin(cost)
    return sigma[index], np.array(cost)[index]


def basic_kernel_regression(x1, y1, x2, y2, n=0.99):
    pca = PCA(n_components=n)
    pca.fit(x1)

    reduced_x1 = pca.transform(x1)
    reduced_x2 = pca.transform(x2)

    sigma, ref_train_error = kernel_learn(reduced_x1, y1)

    y2_ = kernel_regression(reduced_x1, y1, reduced_x2, sigma)
    ref_test_error = np.around(np.mean(np.linalg.norm(y2_ - y2, axis=1)), decimals=3)

    return ref_train_error, ref_test_error, y2_


def siamese_model(input_shape, output_shape, layers):
    initializer = tf.keras.initializers.Identity()

    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Dense(output_shape, activation=None,  use_bias=False, kernel_initializer=initializer))
    for m in range(layers):
        model.add(tf.keras.layers.Dense(output_shape, activation=activation, kernel_initializer=initializer))

    input1 = tf.keras.layers.Input(shape=input_shape)
    input2 = tf.keras.layers.Input(shape=input_shape)

    output1 = model(input1)
    output2 = model(input2)

    l2_layer = tf.keras.layers.Lambda(lambda tensors: K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=False))
    distance = l2_layer([output1, output2])

    siamese = tf.keras.models.Model(inputs=[input1, input2], outputs=[distance, output1, output2])

    return siamese


class TripletLossLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def triplet_model(input_shape, output_shape, layers):
    initializer = tf.keras.initializers.Identity()

    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Dense(output_shape, activation=None,  use_bias=False, kernel_initializer=initializer))
    for m in range(layers):
        model.add(tf.keras.layers.Dense(output_shape, activation=activation, kernel_initializer=initializer))

    input_p = tf.keras.layers.Input(shape=input_shape)
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_n = tf.keras.layers.Input(shape=input_shape)

    output_p = model(input_p)
    output_a = model(input_a)
    output_n = model(input_n)

    output = TripletLossLayer(alpha=0.9, name='triplet_loss_layer')([output_p, output_a, output_n])

    triplet = tf.keras.models.Model(inputs=[input_p, input_a, input_n], outputs=[output_p, output])

    return triplet


def triplet_sampling(x, y):
    N = np.shape(x)[0]
    triplet_indices = np.array(list(permutations(range(N), 3)))
    delete_indices = []
    M = triplet_indices.shape[0]
    for i in range(M):
        indices = triplet_indices[i]
        dist_p = np.linalg.norm(y[indices[0]] - y[indices[1]])
        dist_n = np.linalg.norm(y[indices[2]] - y[indices[1]])
        if dist_p >= dist_n:
            delete_indices.append(i)
    triplet_indices_ = np.delete(triplet_indices, delete_indices, axis=0)

    return x[triplet_indices_], y[triplet_indices_]


def mlkr_loss(d, y, y_true, sigma):
    n = train_size - 1
    m = n * (n + 1)
    Y1 = np.zeros((n + 1, m))
    Y2 = np.zeros((n + 1, m))
    I = np.zeros((n + 1, m))
    y1 = y[:, 0].reshape((n + 1, n))
    y2 = y[:, 1].reshape((n + 1, n))
    kernel = K.exp(-d / sigma)
    loss = 0
    for i in range(train_size):
        Y1[i, i * n: i * n + n] = y1[i]
        Y2[i, i * n: i * n + n] = y2[i]
        I[i, i * n: i * n + n] = 1
        a = tf.keras.layers.Multiply()([Y1[i], kernel])
        b = tf.keras.layers.Multiply()([I[i], kernel])
        y_pred1 = K.sum(a) / K.sum(b)
        a = tf.keras.layers.Multiply()([Y2[i], kernel])
        y_pred2 = K.sum(a) / K.sum(b)
        loss += K.square(y_pred1 - y_true[i, 0]) + K.square(y_pred2 - y_true[i, 1])
    return loss


def nonlinear_metric_learn(x1, x1_, y1, x2, x2_, y2, layers=2, epochs=200, n=0.99):
    pca = PCA(n_components=n)
    pca.fit(x1)
    n = int(pca.n_components_)
    print('Dimension=', n)

    reduced_x1 = pca.transform(x1)
    reduced_x1_ = pca.transform(x1_)
    reduced_x2 = pca.transform(x2)
    reduced_x2_ = pca.transform(x2_)

    pca2 = PCA(n_components=n)
    pca2.fit(reduced_x1_)
    var_ratio = pca2.explained_variance_ratio_
    ref_first_vr = var_ratio[0]

    pca3 = PCA(n_components=n, whiten=True)
    visual_x1 = pca3.fit_transform(reduced_x1)
    visual_x1_ = pca3.transform(reduced_x1_)
    visual_x2 = pca3.transform(reduced_x2)
    visual_x2_ = pca3.transform(reduced_x2_)
    plot_distribution(visual_x1[-exp1_size:], visual_x1_[-exp1_size:], visual_x2, visual_x2_)
    plt.savefig('Distribution.jpg')
    plt.close()

    if SCHEME == 'WSES':
        model = triplet_model(n, n, layers)
    else:
        model = siamese_model(n, n, layers)
    model.summary()

    if SCHEME == 'ES':
        indices = np.triu_indices(x1.shape[0], k=1)
        x_train1 = reduced_x1[indices[0]]
        x_train2 = reduced_x1[indices[1]]
        dist_y1 = pairwise_distances(y1, squared=True)
        y_train_ = dist_y1[indices]
    elif SCHEME == 'IS':
        indices1 = np.zeros(train_size * (train_size - 1), int)
        indices2 = np.zeros(train_size * (train_size - 1), int)
        for i in range(train_size):
            indices1[i * (train_size - 1): (i + 1) * (train_size - 1)] = i
            indices2[i * (train_size - 1): (i + 1) * (train_size - 1)] = np.delete(np.arange(train_size), i)
        x_train1 = reduced_x1[indices1]
        x_train2 = reduced_x1[indices2]
        y_train_ = y1[indices2]
    else:
        x_triplets, y_triplets = triplet_sampling(reduced_x1, y1)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = []

    first_vrs = []

    train_mu = []
    test_mu = []
    min_error = np.inf
    test_error = np.inf

    for epoch in range(epochs):
        if SCHEME == 'WSES':
            x1_transformed, _ = model.predict([reduced_x1, reduced_x1, reduced_x1])
            x1_transformed_, _ = model.predict([reduced_x1_, reduced_x1_, reduced_x1_])
            x2_transformed, _ = model.predict([reduced_x2, reduced_x2, reduced_x2])
            x2_transformed_, _ = model.predict([reduced_x2_, reduced_x2_, reduced_x2_])
        else:
            _, x1_transformed, x1_transformed_ = model.predict([reduced_x1, reduced_x1_])
            _, x2_transformed, x2_transformed_ = model.predict([reduced_x2, reduced_x2_])

        pca2.fit(x1_transformed)
        var_ratio_after = pca2.explained_variance_ratio_
        first_vr = var_ratio_after[0]
        first_vrs.append(first_vr)
        if first_vr >= ref_first_vr:
            if ES:
                print(epoch)
                break

        sigma, train_error = kernel_learn(x1_transformed, y1)
        train_mu.append(train_error)

        y2_ = kernel_regression(x1_transformed, y1, x2_transformed, sigma)
        test_mu.append(np.around(np.mean(np.linalg.norm(y2_ - y2, axis=1)), decimals=3))
        if train_error < min_error:
            min_error = train_error
            test_error = test_mu[-1]
            np.save('Pred_%s.npy' % SCHEME, y2_)

            x1_reduced = pca3.fit_transform(x1_transformed)
            x1_reduced_ = pca3.transform(x1_transformed_)
            x2_reduced = pca3.transform(x2_transformed)
            x2_reduced_ = pca3.transform(x2_transformed_)
            plot_distribution(x1_reduced[-exp1_size:], x1_reduced_[-exp1_size:], x2_reduced, x2_reduced_)
            plt.savefig('Distribution_%s.jpg' % SCHEME)
            plt.close()

            print('gamma=', 1/sigma)

        with tf.GradientTape() as tape:
            if SCHEME == 'ES':
                output1, _, _ = model([x_train1, x_train2], training=True)
                loss = tf.keras.losses.MeanSquaredError()(output1, y_train_)
            elif SCHEME == 'IS':
                output1, _, _ = model([x_train1, x_train2], training=True)
                loss = mlkr_loss(output1, y_train_, y1, sigma)
            else:
                _, output = model([x_triplets[:, 0], x_triplets[:, 1], x_triplets[:, 2]], training=True)
                loss = output
            train_loss.append(loss)

        if epoch != epochs - 1:
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

    training_process(epochs, train_loss, train_mu, test_mu, ref_train_err, ref_test_err, ref_first_vr, first_vrs)
    plt.savefig('Training_%s.jpg' % SCHEME)
    plt.close()

    return min_error, test_error


if __name__ == '__main__':
    activation = 'sigmoid'
    num_layer = 2
    HYBRID = True
    ES = True
    SCHEME = 'IS'
    random_state = 1

    # Load data
    exp1_indices = np.load('exp1_indices.npy')
    exp2_indices = np.load('exp2_indices.npy')
    x_model = np.load('x_model.npy')
    y_model = np.load('y_model.npy')
    x_exp1 = np.load('x_exp1.npy')
    y_exp1 = np.load('y_exp1.npy')
    x_exp2 = np.load('x_exp2.npy')
    y_exp2 = np.load('y_exp2.npy')

    exp1_size = len(exp1_indices)

    x_exp1_ = x_model[exp1_indices - 1]
    x_exp2_ = x_model[exp2_indices - 1]

    if HYBRID:
        x_train = np.concatenate((x_model, x_exp1))
        y_train = np.concatenate((y_model, y_exp1))
        x_train_ = np.concatenate((x_model, x_exp1_))
    else:
        y_train = y_model
        x_train = x_model
        x_train_ = x_model

    x_test, x_test_, y_test = x_exp2, x_exp2_, y_exp2

    train_size = x_train.shape[0]

    # dimension_analysis(np.load('Toy.npy')[np.concatenate((exp1_indices, exp2_indices))-1],
    #                    np.load('Toy_biased.npy')[np.concatenate((exp1_indices, exp2_indices))-1],
    #                    np.load('Toy_noised.npy')[np.concatenate((exp1_indices, exp2_indices))-1])
    # plt.savefig('PCA.jpg')
    # plt.close()

    # plot_damage_locations(y_model, y_exp1, y_exp2)
    # plt.savefig('Real_case.jpg')
    # plt.close()

    # Hyperparameter tuning
    min_err = np.inf
    optimal_energy_ratio = 0.
    ref_train_err = 0
    ref_test_err = 0
    y_pred = y_test
    for energy_ratio in np.arange(0.9, 1., 0.01):
        train_err, test_err, pred = basic_kernel_regression(x_train, y_train, x_test, y_test, n=energy_ratio)
        if train_err < min_err:
            min_err = train_err
            optimal_energy_ratio = energy_ratio
            ref_train_err = train_err
            ref_test_err = test_err
            y_pred = pred
    print(optimal_energy_ratio)
    np.save('Pred_KR.npy', y_pred)

    # Training & testing
    start = time.time()
    train_err, test_err = nonlinear_metric_learn(x_train, x_train_, y_train, x_test, x_test_, y_test,
                                                 layers=num_layer, epochs=300, n=optimal_energy_ratio)
    print(ref_train_err, ref_test_err)
    print(train_err, test_err)
    end = time.time()
    print(end-start)
    # '''
