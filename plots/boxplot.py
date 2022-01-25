import matplotlib.pyplot as plt
import numpy as np

def disp_boxplot(bv_corr, bv_p, mh_corr, mh_p, labels=None, whis=None):
    """ display boxplot of correlations and p_values """
    figure, axis = plt.subplots(1, 2)
    # axis[0].set_ylim([0, 1])
    # axis[1].set_ylim([0, 0.1])
    axis[0].set_title("correlation")
    axis[0].boxplot((bv_corr, mh_corr), whis=whis, labels=labels)
    axis[1].set_title("p_values")
    axis[1].boxplot((bv_p, mh_p), whis=whis, labels=labels)
    plt.show()


def range_boxplot(l_lists, l_params, title='', x_name=''):
    """ plot several boxplots """
    figure, axis = plt.subplots(1, 1)
    axis.set_title(title)
    axis.boxplot(l_lists, labels=l_params)
    plt.xlabel(x_name)
    plt.ylabel('Correlation')
    # plt.legend()
    plt.show()


def draw_curves(l_lists1, l_lists2, l_params, labels=('', ''), title='Average correlation', x_name=''):
    """ plots average correlation for 2 lists of lists of values """
    vals1 = [np.mean(vals) for vals in l_lists1]
    vals2 = [np.mean(vals) for vals in l_lists2]
    plt.plot(l_params, vals1, label=labels[0])
    plt.plot(l_params, vals2, label=labels[1])
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()