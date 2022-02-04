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
    # TODO: code can be improved
    size = math.sqrt(len(l_lists1[0]))
    if hasattr(l_params[0], '__len__'):
        l_params = [str(val) for val in l_params]

    vals1 = np.array([np.mean(vals) for vals in l_lists1])
    vals2 = np.array([np.mean(vals) for vals in l_lists2])
    vals3 = np.array([np.mean(vals) for vals in l_lists3])
    vals4 = np.array([np.mean(vals) for vals in l_lists4])

    range1 = np.array([1.96 * np.std(vals) / size for vals in l_lists1])
    range2 = np.array([1.96 * np.std(vals) / size for vals in l_lists2])
    range3 = np.array([1.96 * np.std(vals) / size for vals in l_lists3])
    range4 = np.array([1.96 * np.std(vals) / size for vals in l_lists4])

    plt.plot(l_params, vals1, label=labels[0])
    plt.fill_between(l_params, vals1 - range1, vals1 + range1, alpha=0.1)
    plt.plot(l_params, vals2, label=labels[1])
    plt.fill_between(l_params, vals2 - range2, vals2 + range2, alpha=0.1)
    plt.plot(l_params, vals3, label=labels[2])
    plt.fill_between(l_params, vals3 - range3, vals3 + range3, alpha=0.1)
    plt.plot(l_params, vals4, label=labels[3])
    plt.fill_between(l_params, vals4 - range4, vals4 + range4, alpha=0.1)

    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel('Correlation')
    plt.legend()
    # plt.show()
    plt.savefig(f'results/{folder}/plot-{title}.png')
