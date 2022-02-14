import matplotlib.pyplot as plt
import numpy as np
import math


def range_boxplot(l_lists, l_params, folder='exp1', title='', x_name=''):
    """ plot several boxplots """
    figure, axis = plt.subplots(1, 1)
    axis.set_title(title)
    axis.boxplot(l_lists, labels=l_params)
    plt.xlabel(x_name)
    plt.ylabel('Correlation')
    # plt.legend()
    # plt.show()
    plt.savefig(f'results/{folder}/boxplot-{title}.png')
    plt.clf()


def draw_curves(l_lists1, l_lists2, l_lists3, l_lists4, l_params, labels=('', '', '', ''), folder='exp1', title='Average correlation', x_name=''):
    """ plots average correlation for 2 lists of lists of values """
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    size = math.sqrt(len(l_lists1[0]))
    if hasattr(l_params[0], '__len__'):
        l_params = [str(val) for val in l_params]
    l_params = np.array(l_params)
    if x_name == "sm":
        x_name = "comparability"

    vals1 = np.array([np.mean(vals) for vals in l_lists1], dtype=object)
    vals2 = np.array([np.mean(vals) for vals in l_lists2], dtype=object)
    vals3 = np.array([np.mean(vals) for vals in l_lists3], dtype=object)
    vals4 = np.array([np.mean(vals) for vals in l_lists4], dtype=object)

    range1 = np.array([1.96 * np.std(vals) / size for vals in l_lists1], dtype=object)
    range2 = np.array([1.96 * np.std(vals) / size for vals in l_lists2], dtype=object)
    range3 = np.array([1.96 * np.std(vals) / size for vals in l_lists3], dtype=object)
    range4 = np.array([1.96 * np.std(vals) / size for vals in l_lists4], dtype=object)

    plt.plot(l_params, vals1, label=labels[0])
    plt.fill_between(list(l_params), list(vals1 - range1), list(vals1 + range1), alpha=0.1)
    plt.plot(l_params, vals2, label=labels[1])
    plt.fill_between(list(l_params), list(vals2 - range2), list(vals2 + range2), alpha=0.1)
    plt.plot(l_params, vals3, label=labels[2])
    plt.fill_between(list(l_params), list(vals3 - range3), list(vals3 + range3), alpha=0.1)
    plt.plot(l_params, vals4, label=labels[3])
    plt.fill_between(list(l_params), list(vals4 - range4), list(vals4 + range4), alpha=0.1)

    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel('Correlation')
    plt.legend()
    # plt.show()
    plt.savefig(f'results/{folder}/plot-{title}.png')
