# TAKEN FROM: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          
                          cmap=None,
                          normalize=False,**kwargs):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

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
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    title=kwargs.get('title','Confusion Matrix')
    accuracy = (np.trace(cm) / float(np.sum(cm)))*100
    misclass = (100 - accuracy)

    # if cmap is None:
        # cmap = plt.get_cmap('YlGnBu')
    # cmap = plt.get_cmap('Greys')
    
    # plt.rcParams.update({'font.size': 20})
    plt.figure(num=None, figsize=(10, 10), dpi=150, facecolor='w', edgecolor='k')    
    plt.imshow(cm, interpolation='nearest', cmap='Greys')
    plt.rcParams.update({'font.size': 15})
    plt.title(title,fontsize=25)
    plt.rcParams.update({'font.size': 20})
    plt.colorbar()

    if target_names is not None:     
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0,fontsize=25)
        plt.yticks(tick_marks, target_names,fontsize=25)


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}%".format(cm[i, j]*100),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white",fontsize=20)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "white",fontsize=20)


    # plt.tight_layout()    
    plt.ylabel('True label',fontsize=25)
    plt.xlabel('\nPredicted label\n Accuracy={:0.2f} %; Error={:0.2f} %'.format(accuracy, misclass),fontsize=25)
    plt.show()