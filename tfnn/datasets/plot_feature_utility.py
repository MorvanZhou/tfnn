import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_feature_utility(data, selected_feature_name, target_name=None):
    """
    This function is to check the categorical feature utility for machine learning BEFORE BINARIZE.
    :param selected_feature_name:
    :param target_name:
    :return:
    """
    if (target_name is None) and (data.ys.shape[1] != 1):
        raise NameError('Target has multiple column, pick one or two')
    target_classes = data.ys[target_name].unique()
    feature_classes = data.xs[selected_feature_name].unique()
    indices = np.arange(len(feature_classes))
    percentages = np.zeros((len(target_classes), len(feature_classes)))
    for j, feature_class in enumerate(feature_classes):
        particular_feature = data.xs[selected_feature_name][data.xs[selected_feature_name] == feature_class]
        feature_total = len(particular_feature)
        for i, target_class in enumerate(target_classes):
            class_count = len(particular_feature[data.ys[target_name] == target_class])
            percentage = class_count / feature_total
            percentages[i, j] = percentage

    colors = ['r', 'b', 'g']
    width = 1
    bars = []
    for i in range(len(target_classes)):
        c_number = int(i % len(colors))
        color = colors[c_number]
        if i == 0:
            bar = plt.bar(indices, percentages[i, :], width, color=color)
        else:
            bar = plt.bar(indices, percentages[i, :], width, color=color, bottom=percentages[:i, :].sum(axis=0))
        bars.append(bar)

    plt.xticks(indices + width / 2, feature_classes)
    plt.ylabel('Percentage')
    plt.xlabel(selected_feature_name)
    plt.legend([bar[0] for bar in bars], target_classes, loc='best')
    plt.show()