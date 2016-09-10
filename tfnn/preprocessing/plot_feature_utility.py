import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_feature_utility(data, n_feature, ):
    """
    This function is to check the categorical feature utility for machine learning BEFORE BINARIZE.
    :param selected_feature_name:
    :return:
    """
    selected_feature = data.xs[:, n_feature]
    target_classes = np.unique(data.ys)
    feature_classes = np.unique(selected_feature)
    indices = np.arange(len(feature_classes))
    percentages = np.zeros((len(target_classes), len(feature_classes)))
    for j, feature_class in enumerate(feature_classes):
        particular_feature = selected_feature[selected_feature == feature_class]
        feature_total = len(particular_feature)
        for i, target_class in enumerate(target_classes):
            class_count = len(particular_feature[data.ys == target_class])
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
    plt.xlabel('Feature_%s' % n_feature)
    plt.legend([bar[0] for bar in bars], target_classes, loc='best')
    plt.show()