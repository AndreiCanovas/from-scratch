import itertools
import numpy as np
import matplotlib.pyplot as plt


def mean_squared_error(y, y_hat):
    mse = (1/len(y))*sum(((y - y_hat)**2))
    return mse

def root_mean_squared_error(y, y_hat):
    rmse = (mean_squared_error(y, y_hat))**(1/2)
    return rmse

def binary_cross_entropy_loss(y, y_hat):
    return -np.mean((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)))

def confusion_matrix(y_true, y_predicted, normalize=True):

    def _plot_confusion_matrix(cm, normalize, target_names=[1, 0],
                               title='Confusion matrix', cmap=None,
                               figsize=(5, 5)):
        """
        given a confusion matrix (cm), make a nice plot

        Arguments
        ---------
        cm:           confusion matrix

        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']

        title:        the text to display at the top of the matrix

        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues

        normalize:    If False, plot the raw numbers
                      If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph

        Citiation
        ---------
        """

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=figsize)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

    results = np.concatenate([y_true, y_predicted], axis=1)

    true_positive  = results[(results[:, 0] == 1) * (results[:, 1] == 1)].shape[0]
    true_negative  = results[(results[:, 0] == 0) * (results[:, 1] == 0)].shape[0]
    false_positive = results[(results[:, 0] == 0) * (results[:, 1] == 1)].shape[0]
    false_negative = results[(results[:, 0] == 1) * (results[:, 1] == 0)].shape[0]

    confusion_matrix = np.array([[true_positive, false_negative],
                                 [false_positive, true_negative]])

    _plot_confusion_matrix(confusion_matrix, normalize=normalize)

    return confusion_matrix