import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def plot_damage_locations(y1, y2, y3):
    plt.rcParams['figure.figsize'] = [6, 2.2]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.subplot(131)
    plt.title('Numerical Model')
    plt.scatter(1 / 6, 1 / 3, c='k', s=10)
    plt.scatter([2 / 3, 5 / 6, 1 / 3], [1 / 6, 2 / 3, 5 / 6], c='gray', s=10)
    for k in range(y1.shape[0]):
        plt.scatter(y1[k, 0], y1[k, 1], c='tab:blue', s=60)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'])
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'], rotation=90)
    plt.grid(linestyle='dotted')
    plt.tight_layout()

    plt.subplot(132)
    plt.title("Surrogate structure")
    plt.scatter(1 / 6, 1 / 3, c='k', s=10)
    plt.scatter([2 / 3, 5 / 6, 1 / 3], [1 / 6, 2 / 3, 5 / 6], c='gray', s=10)
    for k in range(28):
        plt.scatter(y2[k, 0], y2[k, 1], c='tab:red', s=60)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'])
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'], rotation=90)
    plt.grid(linestyle='dotted')
    plt.tight_layout()

    plt.subplot(133)
    plt.title("Monitored structure")
    plt.scatter(1 / 6, 1 / 3, c='k', s=10, label='Actuator')
    plt.scatter([2 / 3, 5 / 6, 1 / 3], [1 / 6, 2 / 3, 5 / 6], c='gray', s=10, label='Sensors')
    for k in range(y3.shape[0]):
        plt.scatter(y3[k, 0], y3[k, 1], c='tab:brown', s=60)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'])
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'], rotation=90)
    plt.grid(linestyle='dotted')
    plt.tight_layout()
    plt.savefig('Damage locations.jpg')
    plt.close()


def show_preds(y1, y1_, y2):
    plt.rcParams['figure.figsize'] = [4.5, 7.5]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    np.set_printoptions(suppress=True, precision=3)

    for i in range(y1.shape[0]):
        plt.subplot(5, 3, i + 1)
        plt.scatter(y2[i, 0], y2[i, 1], s=10, marker='s', c='k', label='Truth')
        plt.scatter(y1[i, 0], y1[i, 1], s=10, c='tab:red', label='EN-MLKR')
        plt.scatter(y1_[i, 0], y1_[i, 1], s=10, c='tab:blue', label='KR')
        plt.xlim(0, 1)
        if i == 12 or i == 13 or i == 14:
            plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'])
        else:
            plt.xticks(np.arange(0, 1.01, 1 / 4), [])
        plt.ylim(0, 1)
        if i == 0 or i == 3 or i == 6 or i == 9 or i == 12:
            plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'], rotation=90)
        else:
            plt.yticks(np.arange(0, 1.01, 1 / 4), [])
        plt.legend()
        plt.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig('Predictions.jpg')
    plt.close()


def kernel_regression(x1, y1, x2, sigma):
    dist = pairwise_distances(x2, x1, squared=True)
    if HYBRID:
        dist = dist[:, :-exp1_size]
        y1 = y1[:-exp1_size]
    softmax = np.exp(- dist / sigma - logsumexp(- dist / sigma, axis=1)[:, np.newaxis])
    y2_ = softmax.dot(y1)
    return y2_


def kernel_learn(x, y):
    dist = pairwise_distances(x, squared=True)
    np.fill_diagonal(dist, np.inf)
    if HYBRID:
        y_ = y
        # dist = np.concatenate((dist, dist[:, :len(simu_indices)]), axis=1)
        # y = np.concatenate((y, y[:len(simu_indices)]))
        dist = dist[:, :-exp1_size]
        y = y[:-exp1_size]
    sigma = np.logspace(-5, -1, 5)
    cost = []
    for k in range(len(sigma)):
        softmax = np.exp(- dist / sigma[k] - logsumexp(- dist / sigma[k], axis=1)[:, np.newaxis])
        yhat = softmax.dot(y)
        if HYBRID:
            ydiff = yhat - y_
        else:
            ydiff = yhat - y
        cost.append(np.mean(np.linalg.norm(ydiff, axis=1)))
    index = np.argmin(cost)
    return sigma[index], np.array(cost)[index]


def basic_kernel_regression(x1, y1, x2, y2, n=0.99):
    pca = PCA(n_components=n)
    pca.fit(x1)

    reduced_x1 = pca.transform(x1)
    reduced_x2 = pca.transform(x2)

    sigma, ref_train_err = kernel_learn(reduced_x1, y1)

    y2_ = kernel_regression(reduced_x1, y1, reduced_x2, sigma)
    np.save('Predictions_.npy', y2_)
    ref_test_err = np.around(np.mean(np.linalg.norm(y2_ - y2, axis=1)), decimals=3)
    ref_test_std = np.around(np.std(np.linalg.norm(y2_ - y2, axis=1)), decimals=3)

    return ref_train_err, ref_test_err, ref_test_std


def siamese_model(input_shape, output_shape, layers):
    initializer = tf.keras.initializers.Identity()

    model = tf.keras.models.Sequential()
    for m in range(layers):
        model.add(tf.keras.layers.Dense(output_shape, activation=activation, kernel_initializer=initializer))

    input1 = tf.keras.layers.Input(shape=input_shape)
    input2 = tf.keras.layers.Input(shape=input_shape)

    output1 = model(input1)
    output2 = model(input2)

    l2_layer = tf.keras.layers.Lambda(lambda tensors: K.sum(K.square(tensors[0] - tensors[1]), axis=1))
    distance = l2_layer([output1, output2])

    siamese = tf.keras.models.Model(inputs=[input1, input2], outputs=[distance, output1, output2])

    return siamese


def nonlinear_metric_learn(x1, x1_, y1, x2, x2_, y2, layers=2, epochs=200, n=0.99):
    pca = PCA(n_components=n)
    pca.fit(x1)
    n = int(pca.n_components_)

    reduced_x1 = pca.transform(x1)
    reduced_x1_ = pca.transform(x1_)
    reduced_x2 = pca.transform(x2)
    reduced_x2_ = pca.transform(x2_)

    pca2 = PCA(n_components=n)
    pca2.fit(reduced_x1_)
    var_ratio = pca2.explained_variance_ratio_
    ref_last_vr = var_ratio[-1]
    pca2.fit(reduced_x1)
    var_ratio_before = pca.explained_variance_ratio_

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

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss = []
    test_loss = []

    last_vrs = []

    train_mu = []
    train_metric = []
    test_mu = []
    test_metric = []
    test_stds = []
    min_err = np.inf

    for epoch in range(epochs):
        _, x1_transformed, x1_transformed_ = model.predict([reduced_x1, reduced_x1_])
        _, x2_transformed, x2_transformed_ = model.predict([reduced_x2, reduced_x2_])

        pca2.fit(x1_transformed)
        var_ratio_after = pca2.explained_variance_ratio_
        last_vr = var_ratio_after[-1]
        last_vrs.append(last_vr)
        if last_vr < ref_last_vr:
            if ES:
                break

        sigma, train_error = kernel_learn(x1_transformed, y1)
        train_mu.append(train_error)

        y2_ = kernel_regression(x1_transformed, y1, x2_transformed, sigma)
        test_mu.append(np.around(np.mean(np.linalg.norm(y2_ - y2, axis=1)), decimals=3))
        test_stds.append(np.around(np.std(np.linalg.norm(y2_ - y2, axis=1)), decimals=3))
        train_metric.append(np.mean(np.linalg.norm(x1_transformed[-exp1_size:] - x1_transformed_[-exp1_size:], axis=1)))
        test_metric.append(np.mean(np.linalg.norm(x2_transformed - x2_transformed_, axis=1)))
        if train_error < min_err:
            min_err = train_error
            np.save('Predictions.npy', y2_)

        with tf.GradientTape() as tape:
            output1, _, _ = model([x_train1, x_train2], training=True)
            loss = loss_fn(output1, y_train_)
            train_loss.append(loss)

            output2, _, _ = model([x_test1, x_test2], training=False)
            test_loss.append(loss_fn(output2, y_test_))

        if epoch != epochs - 1:
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

    plt.rcParams['figure.figsize'] = [6, 2.25]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.subplot(121)
    plt.plot(var_ratio, label='Original modeled data')
    plt.plot(var_ratio_before, label='Original hybrid data')
    plt.plot(var_ratio_after, '--', label='Transformed hybrid data')
    plt.xlabel('Principle components')
    plt.ylabel('Explained variance ratio')
    plt.legend()

    plt.subplot(122)
    plt.plot(train_metric, label='Train')
    plt.plot(test_metric, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Mean distance')
    plt.legend()
    plt.title('Learned distance metric between experiments and models')
    plt.tight_layout()
    plt.savefig('9.jpg')
    plt.close()

    plt.rcParams['figure.figsize'] = [3, 5]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.subplot(311)
    plt.plot(train_loss, 'tab:blue', label='Train')
    # plt.plot(test_loss, 'tab:brown', label='Test')
    plt.ylabel(r'$L_d$')
    plt.legend(loc='upper right')

    plt.subplot(312)
    plt.plot(train_mu, 'tab:blue', label='Train')
    plt.plot(test_mu, 'tab:red', label='Test')
    plt.plot(np.full(epochs, float(ref_train_error)), '--', c='tab:blue', label='Ref_Train')
    plt.plot(np.full(epochs, float(ref_test_error)), '--', c='tab:red', label='Ref_Test')
    plt.ylabel(r'$L_y$')
    plt.legend(loc='upper right')

    plt.subplot(313)
    plt.plot(np.full(epochs, ref_last_vr), label='Original modeled data')
    plt.plot(last_vrs, label='Transformed hybrid data')
    plt.legend(loc='upper right')
    plt.ylabel(r'$\alpha_L$')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('Training process.jpg')
    plt.close()

    return min(train_mu), test_mu[train_mu.index(min(train_mu))], test_stds[train_mu.index(min(train_mu))]


if __name__ == '__main__':

    activation = 'sigmoid'
    HYBRID = True
    ES = False

    # Load data
    exp1_indices = np.array([10, 70, 75, 15, 26, 56, 59, 29, 22, 17, 63, 68, 51, 14,
                             34, 71, 73, 31, 12, 54, 48, 38, 37, 47, 45, 66, 40, 19])
    exp1_size = len(exp1_indices)
    exp2_indices = np.array([10, 17, 37, 11, 13, 25, 30, 39, 41, 44, 46, 55, 65, 67, 74])
    all_indices = np.arange(1, 85, 1)
    simu_indices = np.array(list(set(all_indices) - set(exp1_indices)))

    x_model = np.load('x_model.npy')
    y_model = np.load('y_model.npy')
    x_surro1 = np.load('x_surro1.npy')
    y_surro1 = np.load('y_surro1.npy')
    x_surro2 = np.load('x_surro2.npy')
    y_surro2 = np.load('y_surro2.npy')

    if HYBRID:
        x_train = np.concatenate((x_model[simu_indices - 1], x_model[exp1_indices - 1], x_surro1))
        y_train = np.concatenate((y_model[simu_indices - 1], y_model[exp1_indices - 1], y_surro1))
        x_train_ = np.concatenate((x_model[simu_indices - 1], x_model[exp1_indices - 1], x_model[exp1_indices - 1]))
    else:
        y_train = np.concatenate((y_model[simu_indices - 1], y_surro1))
        x_train = np.concatenate((x_model[simu_indices - 1], x_model[exp1_indices - 1]))
        x_train_ = x_train

    x_test, x_test_, y_test = x_surro2, x_model[exp2_indices - 1], y_surro2

    plot_damage_locations(y_train[:-exp1_size], y_train[-exp1_size:], y_test)

    # Hyperparameter tuning
    min_error = np.inf
    optimal_energy_ratio = 0.
    ref_train_error = 0
    ref_test_error = 0
    ref_test_std_ = 0
    for energy_ratio in np.arange(0.9, 1., 0.01):
        train_err, test_err, test_std = basic_kernel_regression(x_train, y_train, x_test, y_test, n=energy_ratio)
        if train_err < min_error:
            min_error = train_err
            optimal_energy_ratio = energy_ratio
            ref_train_error = train_err
            ref_test_error = test_err
            ref_test_std_ = test_std
    print(optimal_energy_ratio)
    print(ref_train_error, ref_test_error, ref_test_std_)

    # Training & testing
    train_err, test_err, test_std = nonlinear_metric_learn(x_train, x_train_, y_train, x_test, x_test_, y_test,
                                                           layers=2, epochs=100, n=optimal_energy_ratio)
    print(ref_train_error, ref_test_error, ref_test_std_)
    print(train_err, test_err, test_std)

    y_pred = np.load('Predictions.npy')
    y_pred_ = np.load('Predictions_.npy')
    show_preds(y_pred, y_pred_, y_test)
