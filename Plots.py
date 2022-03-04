import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def show_signals(x1, x1_, x2, y2):
    # c = 5342
    t = np.arange(20, 200, 0.2)
    y2 = np.round(y2, decimals=2)

    plt.rcParams['figure.figsize'] = [5.5, 4.1]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.subplot(331)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([2 / 3, 1 / 6]))
    # tof = (s * 0.3 - 0.035) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[0, 0], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[0, 0], c='tab:red', label='Sample structure')
    plt.plot(t, x2[0, 0], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 1' % y2[0])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xticks(np.arange(20, 200, 60), [])

    plt.subplot(332)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([5 / 6, 2 / 3]))
    # tof = (s * 0.3 - 0.01) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[0, 1], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[0, 1], c='tab:red', label='Sample structure')
    plt.plot(t, x2[0, 1], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 2' % y2[0])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xticks(np.arange(20, 200, 60), [])

    plt.subplot(333)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([5 / 6, 2 / 3]))
    # tof = (s * 0.3 - 0.01) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[0, 2], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[0, 2], c='tab:red', label='Sample structure')
    plt.plot(t, x2[0, 2], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 3' % y2[0])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xticks(np.arange(20, 200, 60), [])

    plt.subplot(334)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([2 / 3, 1 / 6]))
    # tof = (s * 0.3 - 0.035) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[9, 0], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[9, 0], c='tab:red', label='Sample structure')
    plt.plot(t, x2[1, 0], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 1' % y2[1])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xticks(np.arange(20, 200, 60), [])

    plt.subplot(335)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([5 / 6, 2 / 3]))
    # tof = (s * 0.3 - 0.01) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[9, 1], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[9, 1], c='tab:red', label='Sample structure')
    plt.plot(t, x2[1, 1], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 2' % y2[1])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xticks(np.arange(20, 200, 60), [])

    plt.subplot(336)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([5 / 6, 2 / 3]))
    # tof = (s * 0.3 - 0.01) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[9, 2], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[9, 2], c='tab:red', label='Sample structure')
    plt.plot(t, x2[1, 2], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 3' % y2[1])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xticks(np.arange(20, 200, 60), [])

    plt.subplot(337)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([2 / 3, 1 / 6]))
    # tof = (s * 0.3 - 0.035) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[22, 0], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[22, 0], c='tab:red', label='Sample structure')
    plt.plot(t, x2[2, 0], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 1' % y2[2])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xlabel(r'Time [$\mu$s]')
    plt.xticks(np.arange(20, 200, 60), np.arange(20, 200, 60))

    plt.subplot(338)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([5 / 6, 2 / 3]))
    # tof = (s * 0.3 - 0.01) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[22, 1], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[22, 1], c='tab:red', label='Sample structure')
    plt.plot(t, x2[2, 1], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 2' % y2[2])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xlabel(r'Time [$\mu$s]')
    plt.xticks(np.arange(20, 200, 60), np.arange(20, 200, 60))

    plt.subplot(339)
    # s = np.linalg.norm(y2[2] - np.array([1 / 6, 1 / 3])) + np.linalg.norm(y2[2] - np.array([5 / 6, 2 / 3]))
    # tof = (s * 0.3 - 0.01) / c * 1e6
    # tof_ix = int(tof / 0.2) - 100
    plt.plot(t, x1_[22, 2], c='tab:blue', label='Numerical model')
    plt.plot(t, x1[22, 2], c='tab:red', label='Sample structure')
    plt.plot(t, x2[2, 2], c='tab:red', alpha=0.5, label='Monitored structure')
    plt.title(r'y=%s - Sensor 3' % y2[2])
    # plt.vlines(t[tof_ix], -1, 1, colors='k', linestyles='dashed')
    plt.xlabel(r'Time [$\mu$s]')
    plt.xticks(np.arange(20, 200, 60), np.arange(20, 200, 60))

    plt.tight_layout()

    plt.savefig('Signal.jpg')
    plt.close()


def dimension_analysis(x_sim, x_exp, x_exp_):
    plt.rcParams['figure.figsize'] = [2.7, 2.7]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    pca = PCA(n_components=29)
    # pca = PCA(n_components=39)
    pca.fit(x_sim)
    var_ratio_sim = pca.explained_variance_ratio_
    pca.fit(x_exp)
    var_ratio_exp = pca.explained_variance_ratio_
    pca.fit(x_exp_)
    var_ratio_exp_ = pca.explained_variance_ratio_

    plt.semilogy(var_ratio_sim, 'tab:blue', label='Modeling data')
    plt.semilogy(var_ratio_exp, 'tab:red', alpha=0.5, label='Biased data')
    plt.semilogy(var_ratio_exp_, 'tab:red', label='Noised data')
    plt.xlabel('Principle components')
    plt.ylabel(r'$\it{\alpha}$')
    plt.legend()
    plt.tight_layout()


def plot_distribution(reduced_x1, reduced_x1_, reduced_x2, reduced_x2_):
    plt.rcParams['figure.figsize'] = [2.7, 2.7]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.scatter(reduced_x1[:, 0], reduced_x1[:, 1], c='tab:red', label='E1')
    plt.scatter(reduced_x1_[:, 0], reduced_x1_[:, 1], c='tab:blue', label='M1')
    plt.scatter(reduced_x2[:, 0], reduced_x2[:, 1], c='tab:red', alpha=0.5, label='E2')
    plt.scatter(reduced_x2_[:, 0], reduced_x2_[:, 1], c='tab:blue', alpha=0.5, label='M2')
    plt.yticks(rotation=90)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.legend()
    plt.tight_layout()

    mu_sim1 = np.mean(reduced_x1_, axis=0)[np.newaxis, :]
    mu_sim2 = np.mean(reduced_x2_, axis=0)[np.newaxis, :]
    mu_exp1 = np.mean(reduced_x1, axis=0)[np.newaxis, :]
    mu_exp2 = np.mean(reduced_x2, axis=0)[np.newaxis, :]
    bias_1 = np.linalg.norm(mu_sim1-mu_exp1)
    bias_2 = np.linalg.norm(mu_sim2-mu_exp2)
    print(bias_1, bias_2)


def plot_distance_map(y1, y2, d, d_):
    plt.rcParams['figure.figsize'] = [3, 3]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    np.set_printoptions(suppress=True, precision=3)
    ix = [3, 7]
    colorbar = 'Reds'

    plt.subplot(2, 2, 1)
    plt.scatter(y1[:, 0], y1[:, 1], s=45, marker='s', c=d[ix[0]], cmap=colorbar)
    plt.scatter(y2[ix[0], 0], y2[ix[0], 1], s=50, marker='s', facecolors='none', edgecolors='tab:red', label='Truth')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'], rotation=90)
    plt.xticks(np.arange(0, 1.01, 1 / 4), [])
    plt.title('Before metric learning')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(y1[:, 0], y1[:, 1], s=45, marker='s', c=d_[ix[0]], cmap=colorbar)
    plt.scatter(y2[ix[0], 0], y2[ix[0], 1], s=50, marker='s', facecolors='none', edgecolors='tab:red', label='Truth')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.01, 1 / 4), [])
    plt.yticks(np.arange(0, 1.01, 1 / 4), [])
    plt.title('After metric learning')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(y1[:, 0], y1[:, 1], s=45, marker='s', c=d[ix[1]], cmap=colorbar)
    plt.scatter(y2[ix[1], 0], y2[ix[1], 1], s=50, marker='s', facecolors='none', edgecolors='tab:red', label='Truth')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'], rotation=90)
    plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'])
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.scatter(y1[:, 0], y1[:, 1], s=45, marker='s', c=d_[ix[1]], cmap=colorbar)
    plt.scatter(y2[ix[1], 0], y2[ix[1], 1], s=50, marker='s', facecolors='none', edgecolors='tab:red', label='Truth')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'])
    plt.yticks(np.arange(0, 1.01, 1 / 4), [])
    plt.legend()

    plt.tight_layout()


def show_preds(y0, y1, y2, y3, y4):
    plt.rcParams['figure.figsize'] = [5.5, 5.5]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 8
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    np.set_printoptions(suppress=True, precision=3)

    for i in range(y1.shape[0]):
        plt.subplot(4, 4, i + 2)
        plt.scatter(y0[i, 0], y0[i, 1], s=10, c='k', marker='s', label='Truth')
        plt.scatter(y1[i, 0], y1[i, 1], s=10, c='tab:red', marker='o', label=r'Without $\varphi$')
        plt.scatter(y2[i, 0], y2[i, 1], s=10, marker='^', c='tab:blue', label='ES')
        plt.scatter(y3[i, 0], y3[i, 1], s=10, marker='>', c='tab:blue', label='WSES')
        plt.scatter(y4[i, 0], y4[i, 1], s=10, marker='<', c='tab:blue', label='IS')
        plt.xlim(0, 1)
        if i == 11 or i == 12 or i == 13 or i == 14:
            plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'])
        else:
            plt.xticks(np.arange(0, 1.01, 1 / 4), [])
        plt.ylim(0, 1)
        if i == 0 or i == 3 or i == 7 or i == 11:
            plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '0.25', '0.5', '0.75', '1'], rotation=90)
        else:
            plt.yticks(np.arange(0, 1.01, 1 / 4), [])
        # if i == 0:
        #     plt.legend()
        plt.grid(linestyle=':')
    plt.tight_layout()
    plt.savefig('Predictions.jpg')
    plt.close()


def plot_damage_locations(y0, y1, y2):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.figsize'] = [3.5, 3.5]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.grid(linestyle='dotted')
    plt.scatter(1 / 6, 1 / 3, c='k', s=10)
    plt.scatter([2 / 3, 5 / 6, 1 / 3], [1 / 6, 2 / 3, 5 / 6], c='gray', s=10)
    for k in range(y0.shape[0]):
        if k == 0:
            plt.scatter(y0[k, 0], y0[k, 1], facecolors='none', edgecolors='tab:blue', marker='o', s=70, label='Modeling training set')
        else:
            plt.scatter(y0[k, 0], y0[k, 1], facecolors='none', edgecolors='tab:blue', marker='o', s=70)
    for k in range(y1.shape[0]):
        if k == 0:
            plt.scatter(y1[k, 0], y1[k, 1], facecolors='none', edgecolors='tab:red', marker='o', s=27, label='Experimental training set')
        else:
            plt.scatter(y1[k, 0], y1[k, 1], facecolors='none', edgecolors='tab:red', marker='o', s=27)
    for k in range(y2.shape[0]):
        if k == 0:
            plt.scatter(y2[k, 0], y2[k, 1], facecolors='tab:red', edgecolors='none', marker='o', s=7, label='Experimental testing set')
        else:
            plt.scatter(y2[k, 0], y2[k, 1], facecolors='tab:red', edgecolors='none', marker='o', s=7)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.01, 1 / 4), ['0', '75', '150', '225', '300'])
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 1 / 4), ['0', '75', '150', '225', '300'], rotation=90)
    plt.xlabel('Coordinate 1 [mm]')
    plt.ylabel('Coordinate 2 [mm]', rotation=90)

    plt.legend(loc='upper right')
    plt.tight_layout()


def training_process(epochs, train_loss, train_mu, test_mu, ref_train_err, ref_test_err, ref_last_vr, last_vrs):
    plt.rcParams['figure.figsize'] = [2.4, 4]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.subplot(311)
    plt.plot(train_loss, 'tab:blue', label='Train')
    plt.ylabel(r'Loss')
    plt.legend()
    plt.xlim(0, epochs)
    plt.xticks(np.arange(0, epochs, int(epochs / 4)), [])
    plt.tight_layout()

    plt.subplot(312)
    plt.plot(train_mu, 'tab:blue', label='Train')
    plt.plot(test_mu, 'tab:red', label='Test')
    plt.plot(np.full(epochs, float(ref_train_err)), '--', c='tab:blue', label='Ref_Train')
    plt.plot(np.full(epochs, float(ref_test_err)), '--', c='tab:red', label='Ref_Test')
    min_indx = np.argmin(train_mu)
    plt.plot(min_indx, train_mu[min_indx], 'ks')
    plt.annotate(str(round(train_mu[min_indx], 3)), xy=[min_indx, train_mu[min_indx]])
    plt.plot(min_indx, test_mu[min_indx], 'ks')
    plt.annotate(str(round(test_mu[min_indx], 3)), xy=[min_indx, test_mu[min_indx]])
    plt.ylabel(r'Error')
    plt.legend()
    plt.xlim(0, epochs)
    plt.xticks(np.arange(0, epochs, int(epochs / 4)), [])
    plt.tight_layout()

    plt.subplot(313)
    plt.plot(np.full(epochs, ref_last_vr), 'tab:blue', label=r'$\it{Z_m}$')
    plt.plot(last_vrs, 'tab:red', label=r'$\it{\phi(Z)}$')
    plt.legend()
    plt.ylabel(r'$\it{\alpha_1}$')
    plt.xlabel('Epoch')
    plt.xlim(0, epochs)
    plt.xticks(np.arange(0, epochs, int(epochs / 4)), np.arange(0, epochs, int(epochs / 4)))
    plt.tight_layout()

