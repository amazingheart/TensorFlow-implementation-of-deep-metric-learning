import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def cosine_similarity(exp, simu):
    exp /= np.linalg.norm(exp, axis=1)[:, np.newaxis]
    simu /= np.linalg.norm(simu, axis=1)[:, np.newaxis]
    return np.trace(exp.dot(simu.T))/exp.shape[0]


def kernel_regression(x1, y1, x2, sigma):
    dist = pairwise_distances(x2, x1, squared=True)
    softmax = np.exp(- dist / sigma - logsumexp(- dist / sigma, axis=1)[:, np.newaxis])
    y2_ = softmax.dot(y1)
    return y2_


def kernel_learn(exp_size, x, y):
    dist = pairwise_distances(x, squared=True)

    np.fill_diagonal(dist, np.inf)
    sigma = np.logspace(-7, 3, 10)
    cost = []
    cost2 = []
    for k in range(len(sigma)):
        softmax = np.exp(- dist / sigma[k] - logsumexp(- dist / sigma[k], axis=1)[:, np.newaxis])
        yhat = softmax.dot(y)
        ydiff = yhat - y
        cost.append(np.mean(np.linalg.norm(ydiff, axis=1)))
        cost2.append(np.mean(np.linalg.norm(ydiff, axis=1)[-exp_size:]))

    index = np.argmin(cost)

    # plt.semilogx(sigma, cost)
    # plt.title('Min MND = %.3f when var = %.1e' % (np.min(cost), sigma[index]))
    # plt.xlabel('Deviation')
    # plt.ylabel('MND')
    # plt.show()

    # softmax = np.exp(- dist / sigma[index] - logsumexp(- dist / sigma[index], axis=1)[:, np.newaxis])
    # for k in np.arange(0, y.shape[0], 4):
    #     plt.plot(np.sort(softmax[k]))
    # plt.xlabel('Neighbours')
    # plt.ylabel('Kernel')
    # plt.show()

    return sigma[index], cost2[int(index)]


def siamese_model(input_shape, output_shape, layers):
    initializer = tf.keras.initializers.Identity()

    model = tf.keras.models.Sequential()
    for m in range(layers):
        model.add(tf.keras.layers.Dense(output_shape, activation='tanh', kernel_initializer=initializer))

    input1 = tf.keras.layers.Input(shape=input_shape)
    input2 = tf.keras.layers.Input(shape=input_shape)

    output1 = model(input1)
    output2 = model(input2)

    l2_layer = tf.keras.layers.Lambda(lambda tensors: K.sum(K.square(tensors[0] - tensors[1]), axis=1))
    distance = l2_layer([output1, output2])

    siamese = tf.keras.models.Model(inputs=[input1, input2], outputs=[distance, output1, output2])

    return siamese


def nonlinear_metric_learn(exp_size, signal_length, x1, x1_, y1, x2, x2_, y2, lr=1e-3, layers=2, epochs=200, n=30):
    pca = PCA(n_components=n)
    pca.fit(x1)

    reduced_x1 = pca.transform(x1)
    reduced_x1_ = pca.transform(x1_)
    reduced_x2 = pca.transform(x2)
    reduced_x2_ = pca.transform(x2_)

    pca2 = PCA(n_components=n)
    pca2.fit(reduced_x1_)
    var_ratio = pca2.explained_variance_ratio_
    ref_last_vr = var_ratio[-1]

    sigma, ref_train_error = kernel_learn(exp_size, reduced_x1, y1)

    y2_ = kernel_regression(reduced_x1, y1, reduced_x2, sigma)
    ref_test_error = np.around(np.mean(np.linalg.norm(y2_ - y2, axis=1)), decimals=3)

    model = siamese_model(n, n, layers)
    model.summary()

    indices = np.triu_indices(x1.shape[0], k=1)
    x_train1 = tf.convert_to_tensor(reduced_x1[indices[0]])
    x_train2 = tf.convert_to_tensor(reduced_x1[indices[1]])
    dist_y1 = pairwise_distances(y1, squared=True)
    y_train_ = tf.convert_to_tensor(dist_y1[indices])

    indices2 = np.indices((x1.shape[0], x2.shape[0]))
    x_test1 = tf.convert_to_tensor(reduced_x1[indices2[0].ravel()])
    x_test2 = tf.convert_to_tensor(reduced_x2[indices2[1].ravel()])
    dist_y2 = pairwise_distances(y1, y2, squared=True)
    y_test_ = tf.convert_to_tensor(dist_y2.ravel())

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss = []
    test_loss = []

    last_vrs = []

    train_mu = []
    test_mu = []

    for epoch in range(epochs):
        _, x1_transformed, x1_transformed_ = model.predict([reduced_x1, reduced_x1_])
        _, x2_transformed, x2_transformed_ = model.predict([reduced_x2, reduced_x2_])

        pca2.fit(x1_transformed)
        var_ratio = pca2.explained_variance_ratio_
        last_vr = var_ratio[-1]
        last_vrs.append(last_vr)
        if last_vr < ref_last_vr:
            if epoch == 0:
                return ref_test_error, ref_test_error
            break

        sigma, train_error = kernel_learn(exp_size, x1_transformed, y1)
        train_mu.append(train_error)

        y2_ = kernel_regression(x1_transformed, y1, x2_transformed, sigma)
        test_mu.append(np.around(np.mean(np.linalg.norm(y2_ - y2, axis=1)), decimals=3))

        with tf.GradientTape() as tape:
            output1, _, _ = model([x_train1, x_train2], training=True)
            loss = loss_fn(output1, y_train_)
            train_loss.append(loss)

            output2, _, _ = model([x_test1, x_test2], training=False)
            test_loss.append(loss_fn(output2, y_test_))

        if epoch != epochs - 1:
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # plt.plot(train_mu, 'k', label='Train')
    # plt.plot(test_mu, 'r', label='Test')
    # plt.plot(np.full(epochs, float(ref_train_error)), '--k', label='Ref_Train_exp')
    # plt.plot(np.full(epochs, float(ref_test_error)), '--r', label='Ref_Test')
    # plt.legend()
    # plt.title('MND - %s length - %s layers - %d reserved dimension' % (signal_length, layers, n))
    # plt.tight_layout()
    # plt.show()
    #
    # plt.plot(np.full(epochs, ref_last_vr), label='PCA of the original modeled data')
    # plt.plot(last_vrs, label='PCA of the transformed data')
    # plt.legend()
    # plt.title('The variance ratio on the last principle component')
    # plt.show()
    #
    # plt.plot(train_loss)
    # plt.plot(test_loss)
    # plt.legend(['Train', 'Test'])
    # plt.title('Loss')
    # plt.tight_layout()
    # plt.show()

    return test_mu[train_mu.index(min(train_mu))], ref_test_error
