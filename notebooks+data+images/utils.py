import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
def model_visualization(model,X,y,classifier):
    """
    Takes results from training dataset & visualizes it using ListedColormap
    :param model: name of the model to print on top of visual.
    :param X: train or test x predictors
    :param y: train or test y label
    :return: It returns a plot. The image is not saved.
    """
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X, y
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.6, cmap = ListedColormap(('red', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    color = ListedColormap(('red', 'blue'))(i), label = j)
    plt.title("%s Model Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.savefig('images/{0}.png'.format(model))  

def data_plot(hue, data):
    """
    Takes pandas dataframe, creates a countplot, print plots to image folder
    :param data: pandas dataframe column(s) to be visualized
    :param hue: label for prediction from pandas dataframe (extracted as an array)
    :return: It returns a plot. Expected to read by pandas dataframe.
    """
    for i, col in enumerate(data.columns):
        plt.figure(i)
        sns.set(rc={'figure.figsize':(7, 3)})
        sns.countplot(x=data[col],palette='husl',hue=hue,data=data)
        plt.savefig('images/{0}.png'.format(col))       

from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def print_score(classifier,X_test,y_test):
    """
    Takes in classifier, x & y test variables and print the model's accuracy
    classification report, and confusion matrix
    :param classifier: classifier that the model has been sustantiated
    :param X_test: test predictors
     :param y_test: test labels
    :return: printed output of accuracy score, classification report, and 
    confusion matrix.
    """
    print("Test results:\n")
    print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
    print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))
    print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))