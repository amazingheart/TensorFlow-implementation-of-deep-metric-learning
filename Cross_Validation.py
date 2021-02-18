import numpy as np
import matplotlib.pyplot as plt
from Nonlinear_MLKR_Siamese import nonlinear_metric_learn


def screen(x, window):
    return x[:, :, :window].reshape((x.shape[0], -1))


def splitting(ix):
    fold1 = ix[:9]
    fold2 = ix[9:18]
    fold3 = ix[18:]
    train1 = np.concatenate((fold2, fold3))
    train2 = np.concatenate((fold1, fold3))
    train3 = np.concatenate((fold1, fold2))
    return fold1, fold2, fold3, train1, train2, train3


if __name__ == '__main__':
    exp1_indices = np.array([10, 12, 14, 15, 17, 19, 22, 26, 29, 31, 34, 38, 40,
                             45, 51, 47, 54, 56, 59, 63, 66, 68, 70, 71, 73, 75])

    repetition1 = np.argsort(np.argsort([10, 15, 19, 38, 40, 45, 66, 70, 75, 14, 17, 22, 34,
                                         47, 51, 63, 68, 71, 12, 26, 29, 31, 54, 56, 59, 73]))
    repetition2 = np.argsort(np.argsort([17, 19, 22, 40, 45, 47, 63, 66, 68, 12, 14, 31, 34,
                                         38, 51, 54, 71, 73, 10, 15, 26, 29, 56, 59, 70, 75]))
    repetition3 = np.argsort(np.argsort([10, 15, 19, 40, 45, 47, 66, 70, 75, 12, 17, 22, 31,
                                         38, 54, 63, 68, 73, 14, 26, 29, 34, 51, 56, 59, 71]))
    repetition4 = np.argsort(np.argsort([14, 17, 22, 34, 38, 51, 63, 68, 71, 12, 19, 31, 40,
                                         45, 47, 54, 66, 73, 10, 15, 26, 29, 56, 59, 70, 75]))
    r1v1, r1v2, r1v3, r1t1, r1t2, r1t3 = splitting(repetition1)
    r2v1, r2v2, r2v3, r2t1, r2t2, r2t3 = splitting(repetition2)
    r3v1, r3v2, r3v3, r3t1, r3t2, r3t3 = splitting(repetition3)
    r4v1, r4v2, r4v3, r4t1, r4t2, r4t3 = splitting(repetition4)
    valid_ids = [[r1v1, r1v2, r1v3],
                 [r2v1, r2v2, r2v3],
                 [r3v1, r3v2, r3v3],
                 [r4v1, r4v2, r4v3]]
    train_ids = [[r1t1, r1t2, r1t3],
                 [r2t1, r2t2, r2t3],
                 [r3t1, r3t2, r3t3],
                 [r4t1, r4t2, r4t3]]

    y_simu = np.load('y_simu.npy')
    y_exp1 = np.load('y_exp1.npy')

    errs = []
    ref_errs = []

    for signal_length in [600, 650, 700]:
        x_simu = screen(np.load('x_simu.npy'), signal_length)
        x_exp1 = screen(np.load('x_exp1.npy'), signal_length)
        for theta in np.arange(20, 40, 1):
            for num_layers in [1, 2, 3]:
                count = 0
                err = 0
                ref_err = 0
                for r in range(4):
                    for f in range(3):
                        train_indices = train_ids[r][f]
                        valid_indices = valid_ids[r][f]
                        exp_size = len(train_indices)
                        x_train = np.concatenate((x_simu, x_exp1[train_indices]))
                        y_train = np.concatenate((y_simu, y_exp1[train_indices]))
                        x_valid = x_exp1[valid_indices]
                        y_valid = y_exp1[valid_indices]

                        id1 = exp1_indices[train_indices] - 1
                        x_train_ = np.concatenate((x_simu, x_simu[id1]))
                        id2 = exp1_indices[valid_indices] - 1
                        x_valid_ = x_simu[id2]

                        err1, relative_err1 = nonlinear_metric_learn(exp_size, signal_length,
                                                                     x_train, x_train_, y_train,
                                                                     x_valid, x_valid_, y_valid,
                                                                     lr=1e-3, layers=num_layers,
                                                                     epochs=500, n=int(theta))
                        err += err1
                        ref_err += relative_err1
                        count += 1
                errs.append(err / count)
                ref_errs.append(ref_err / count)

    plt.plot(errs)
    plt.plot(ref_errs)
    plt.xticks(np.arange(0, 181, 3))
    plt.grid()
    plt.title('3-fold cross validation')
    plt.show()
