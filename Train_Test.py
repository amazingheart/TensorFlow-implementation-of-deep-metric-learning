import numpy as np
from Nonlinear_MLKR_Siamese import nonlinear_metric_learn


def screen(x, window):
    return x[:, :, :window].reshape((x.shape[0], -1))


if __name__ == '__main__':
    exp1_indices = np.array([10, 12, 14, 15, 17, 19, 22, 26, 29, 31, 34, 38, 40,
                             45, 51, 47, 54, 56, 59, 63, 66, 68, 70, 71, 73, 75])
    exp2_indices = np.array([11, 13, 25, 30, 37, 39, 41, 44, 46, 55, 65, 67, 74])
    exp_size = len(exp1_indices)
    y_train = np.concatenate((np.load('y_simu.npy'), np.load('y_exp1.npy')))
    y_test = np.load('y_exp2.npy')

    for signal_length in [650]:
        x_simu = screen(np.load('x_simu.npy'), signal_length)
        x_exp1 = screen(np.load('x_exp1.npy'), signal_length)
        x_train = np.concatenate((x_simu, x_exp1))
        x_test = screen(np.load('x_exp2.npy'), signal_length)
        x_train_ = np.concatenate((x_simu, x_simu[exp1_indices - 1]))
        x_test_ = x_simu[exp2_indices - 1]
        for theta in [33]:
            for num_layers in [1]:
                err1, ref_err1 = nonlinear_metric_learn(exp_size, signal_length,
                                                        x_train, x_train_, y_train,
                                                        x_test, x_test_, y_test,
                                                        lr=1e-3, layers=num_layers,
                                                        epochs=500, n=int(theta))
