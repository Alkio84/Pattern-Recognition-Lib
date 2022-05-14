import matplotlib.pyplot as plt


def plot_multiple_bayes_error(S_list, labels, logPriors, legend=None):
    for i, S in enumerate(S_list):
        legend_i = legend[i] if legend is not None and i < len(legend) else None

        dcfs, min_dcfs = bayes_errors_from_priors(S, labels, logPriors)
        last_plot = plt.plot(logPriors, dcfs, label=f"DCF {legend_i}")
        # Print the min dcf with a line of the same color as the dcf but with a dashed style
        plt.plot(logPriors, min_dcfs, label=f"minDCF {legend_i}", color=last_plot[0].get_color(), linestyle="dashed")
    if legend is not None:
        plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([min(logPriors), max(logPriors)])
    plt.xlabel("log(pt/(1-pt))")
    plt.ylabel("DCF")
    plt.show()


def plot_multiple_mindcf_bar_chart(minDCFs, x, legend=None, x_label=None):
    ax = plt.subplot()
    width = 1 / (len(minDCFs) + 1)  # One bar as padding
    for i, minDCF in enumerate(minDCFs):
        legend_i = legend[i] if legend is not None and i < len(legend) else None    # Name of the model

        ax.bar(x+(width*i), minDCF, width=width, label=f"{legend_i}")  # minDCF is a vector, minDCFs is a list of vectors
    if x_label is not None:
        plt.xlabel(x_label)    # legend_x is the x-axis label
    if legend is not None:
        plt.legend()
    plt.rcParams['text.usetex'] = True
    plt.xlabel(x_label)
    plt.ylabel("minDCF")
    plt.show()


def plot_multiple_mindcf_chart(minDCFs, x, legend=None, x_label=None, x_scale=None):
    ax = plt.subplot()
    for i, minDCF in enumerate(minDCFs):
        legend_i = legend[i] if legend is not None and i < len(legend) else None    # Name of the model

        ax.plot(x, minDCF, label=f"{legend_i}")  # minDCF is a vector, minDCFs is a list of vectors
    if x_label is not None:
        plt.xlabel(x_label)    # legend_x is the x-axis label
    if legend is not None:
        plt.legend()
    if x_scale is not None:
        plt.xscale(x_scale)
    plt.xlabel(x_label)
    plt.ylabel("DCF")
    plt.show()

