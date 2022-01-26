import matplotlib.pyplot as plt
import numpy as np
import math

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


def draw_curves(l_lists1, l_lists2, l_lists3, l_params, labels=('', '', ''), title='Average correlation', x_name=''):
    """ plots average correlation for 2 lists of lists of values """
    size = math.sqrt(len(l_lists1[0]))

    vals1 = np.array([np.mean(vals) for vals in l_lists1])
    vals2 = np.array([np.mean(vals) for vals in l_lists2])
    vals3 = np.array([np.mean(vals) for vals in l_lists3])


    range1 = np.array([1.96 * np.std(vals1) / size for vals in l_lists1])     
    range2 = np.array([1.96 * np.std(vals2) / size for vals in l_lists2])
    range3 = np.array([1.96 * np.std(vals3) / size for vals in l_lists3])
    

    plt.plot(l_params, vals1, label=labels[0])
    plt.fill_between(l_params, vals1 - range1, vals1 + range1, alpha=0.1)
    plt.plot(l_params, vals2, label=labels[1])
    plt.fill_between(l_params, vals2 - range2, vals2 + range2, alpha=0.1)
    plt.plot(l_params, vals3, label=labels[2])
    plt.fill_between(l_params, vals3 - range3, vals3 + range3, alpha=0.1)

    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()