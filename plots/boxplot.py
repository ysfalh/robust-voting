import matplotlib.pyplot as plt


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
