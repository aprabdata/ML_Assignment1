import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


from ucimlrepo import fetch_ucirepo
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


from sklearn.svm import SVC
from sklearn.model_selection import LearningCurveDisplay, learning_curve, ShuffleSplit
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.utils import shuffle

from sklearn.neighbors import KNeighborsClassifier



#Iris Dataset
# fetch Iris dataset
def fetch_iris_data():
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    X = iris.data.features
    y = iris.data.targets

    # metadata
    print(iris.metadata)

    # variable information
    print(iris.variables)
    return X, y

#Wine Dataset
# fetch wine dataset
def fetch_wine_data():
    wine = fetch_ucirepo(id=109)

    # data (as pandas dataframes)
    X = wine.data.features
    y = wine.data.targets

    # metadata
    print(wine.metadata)

    # variable information
    print(wine.variables)
    return X, y

params = [
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "adaptive",
        "momentum": 0,
        "learning_rate_init": 0.2,
    },
    {
        "solver": "sgd",
        "learning_rate": "constant",
        "momentum": 0,
        "learning_rate_init": 0.3,
    },
    {
        "solver": "sgd",
        "learning_rate": "adaptive",
        "momentum": 0,
        "learning_rate_init": 0.3,
    },
]

plot_args_val = [
    {"c":"green", "linestyle":"solid"},
    {"c":"green", "linestyle": "dotted"},
    {"c":"green", "linestyle": "dashed"},
    {"c":"green", "linestyle": "dashdot"},
]
plot_args_loss = [
    {"c":"red", "linestyle":"solid"},
    {"c":"red", "linestyle": "dotted"},
    {"c":"red", "linestyle": "dashed"},
    {"c":"red", "linestyle": "dashdot"},
]

def apply_mlp(X, y, data_name):


    encoder = OneHotEncoder(sparse=False)

    X_normalized = MinMaxScaler().fit_transform(X)
    y_encoded = encoder.fit_transform(y)

    mlps = []

    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded,
                                                                            train_size=0.8, random_state=40,
                                                                            stratify=y_encoded)



    hidden_layers_single_low = [(10), (20), (30), (40)]
    #hidden_layers_single_high = [(100), (150),(200), (250)]
    hidden_layers_double_low = [(10,10), (20, 20), (30, 30), (40,40)]
    #hidden_layers_double_high = [(100, 100), (150, 150), (200, 200), (250, 250)]

    """
    for param in params:
        mlp = MLPClassifier(random_state=1, max_iter=3000, early_stopping=True, validation_fraction=0.2, **param)
        mlp.fit(X_train, y_train)
        #print("Training set score: %f" % mlp.score(X_train, y_train))
        #print("Test set score: %f" % mlp.score(X_test, y_test))
        #print("Training set loss: %f" % mlp.loss_)
        mlps.append(mlp)

    """
    # Single Layer
    fig_learn, axes_learn = plt.subplots(2, 2, figsize=(8,10.5))
    #fig_learn.suptitle(f"MLP with Single Hidden Layer: Loss and Validation Curve for {data_name}", fontsize=12)
    labels_validation_low = [
        "Validation: 10units/layer",
        "Validation: 20units/layer",
        "Validation: 30units/layer",
        "Validation: 40units/layer",
    ]
    labels_loss_low = [
        "Loss: 10units/layer",
        "Loss: 20units/layer",
        "Loss: 30units/layer",
        "Loss: 40units/layer",
    ]
    labels_all_low = [
        "Validation: 10units/layer",
        "Loss: 10units/layer",
        "Validation: 20units/layer",
        "Loss: 20units/layer",
        "Validation: 30units/layer",
        "Loss: 30units/layer",
        "Validation: 40units/layer",
        "Loss: 40units/layer",
    ]

    for ax, param in zip(axes_learn.ravel(), params):
        for hidden_layers, plot_arg_val, plot_arg_loss, label_validation, label_loss in zip(hidden_layers_single_low, plot_args_val, plot_args_loss, labels_validation_low, labels_loss_low):
            mlp = MLPClassifier(random_state=1, max_iter=3000, early_stopping=True, hidden_layer_sizes= hidden_layers, validation_fraction=0.2, **param)
            mlp.fit(X_train, y_train)
            ax.plot(mlp.validation_scores_, label= label_validation, **plot_arg_val)
            ax.plot(mlp.loss_curve_, label= label_loss, **plot_arg_loss)
            #ax.legend()
            ax.set_xlabel('epoch')
            ax.set_title(f"Learning Rate: {mlp.learning_rate}, {mlp.learning_rate_init}")
        #fig_learn.legend(ax.get_lines(), labels_validation_low, ncol=2, loc="lower center")
        #fig_learn.legend(ax.get_lines(), labels_loss_low, ncol=2, loc= "upper center")
        fig_learn.legend(ax.get_lines(), labels_all_low, ncol=2, loc="lower center")
            #ax.text(3, 2, f"Training set score: {mlp.score(X_train, y_train)}")
    plt.savefig('MLP_single_'+data_name+'.png')
    plt.clf()

    ### Double Layer
    fig_learn, axes_learn = plt.subplots(2, 2, figsize=(8, 10.5))
    # fig_learn.suptitle(f"MLP with Single Hidden Layer: Loss and Validation Curve for {data_name}", fontsize=12)

    for ax, param in zip(axes_learn.ravel(), params):
        for hidden_layers, plot_arg_val, plot_arg_loss, label_validation, label_loss in zip(hidden_layers_double_low,
                                                                                            plot_args_val,
                                                                                            plot_args_loss,
                                                                                            labels_validation_low,
                                                                                            labels_loss_low):
            mlp = MLPClassifier(random_state=1, max_iter=3000, early_stopping=True, hidden_layer_sizes=hidden_layers,
                                validation_fraction=0.2, **param)
            mlp.fit(X_train, y_train)
            ax.plot(mlp.validation_scores_, label=label_validation, **plot_arg_val)
            ax.plot(mlp.loss_curve_, label=label_loss, **plot_arg_loss)
            # ax.legend()
            ax.set_xlabel('epoch')
            ax.set_title(f"Learning Rate: {mlp.learning_rate}, {mlp.learning_rate_init}")
        fig_learn.legend(ax.get_lines(), labels_all_low, ncol=2, loc="lower center")

    plt.savefig('MLP_double_' + data_name + '.png')
    plt.clf()
    """
    common_params = {
        "X": X_normalized,
        "y": y_encoded,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=5, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": None,
        "score_name": "Accuracy",
    }
    
    for param in params:
        mlp = MLPClassifier(random_state=1, max_iter=10, early_stopping=False, **param)
        mlps.append(mlp)

    for ax, mlp in zip(axes.ravel(), mlps):
        LearningCurveDisplay.from_estimator(mlp, **common_params, ax=ax)
        handles, label = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["Training Score", "Test Score"])
        ax.set_title(f"Learning Curve for {mlp.__class__.__name__}")
    """
    """
    #Validation Curve for parameter optimization
    fig_val, axes_val = plt.subplots(1, 2, figsize=(4, 5))
    alphas = np.logspace(-5, -1, 5)
    X, y = shuffle(X_normalized, y_encoded, random_state=0)

    mlp = MLPClassifier(random_state=1, max_iter=1000, early_stopping=True, hidden_layer_sizes= (20), solver='sgd', learning_rate = 'adaptive' )
    ValidationCurveDisplay.from_estimator(mlp, X, y, param_name="alpha", param_range=alphas, ax=axes_val[0, 0], std_display_style = None, score_name= "Accuracy")
    axes_val[0, 0].set_title(f"Hidden Layers: {mlp.hidden_layer_sizes}, Solver: {mlp.solver}, Learning Rate: {mlp.learning_rate}")

    mlp = MLPClassifier(random_state=1, max_iter=1000, early_stopping=True, hidden_layer_sizes=(30), solver='sgd', learning_rate = 'adaptive')
    ValidationCurveDisplay.from_estimator(mlp, X, y, param_name="alpha", param_range=alphas, ax=axes_val[0, 1],std_display_style=None, score_name="Accuracy")
    axes_val[0, 1].set_title(f"Hidden Layers: {mlp.hidden_layer_sizes}, Solver: {mlp.solver}, Learning Rate: {mlp.learning_rate}")

    plt.savefig('MLP_validation_' + data_name + '.png')
    #plt.show()
    plt.clf()
    """



def apply_svm(X, y, data_name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 10))
    encoder = OneHotEncoder(sparse=False)

    X_normalized = MinMaxScaler().fit_transform(X)
    y_encoded = encoder.fit_transform(y)

    common_params = {
        "X": X_normalized,
        "y": y_encoded,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=5, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": None,
        "score_name": "Accuracy",
    }

    svcs = []
    svc_rbf = SVC(kernel="rbf", gamma=0.001)
    svcs.append(svc_rbf)
    svc_poly= SVC(kernel="poly", gamma= 0.01, degree=10)
    svcs.append(svc_poly)
    #svc_sigmoid = SVC(kernel="sigmoid", gamma=0.001, C= 0.00000001)
    #svcs.append(svc_sigmoid)


    for ax, svc in zip(axes.ravel(), svcs):
        LearningCurveDisplay.from_estimator(svc, **common_params, ax=ax)
        handles, label = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["Training Score", "Test Score"])
        ax.set_title(f"Learning Curve for {svc.__class__.__name__}")

    """
    gammas = np.logspace(-5, -1, 5)
    X, y = shuffle(X_normalized, y_encoded, random_state=0)
    svc = SVC(kernel="rbf", gamma=0.001)
    ValidationCurveDisplay.from_estimator(svc, X, y, param_name="gamma",
                                          param_range=gammas, ax=axes[0], std_display_style=None,
                                          score_name="Accuracy")
    axes[0].set_title(f"Weights function used: {svc.weights}")
    """
    plt.show()

def apply_knn(X, y, data_name):
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y)

    X_normalized = MinMaxScaler().fit_transform(X)
    """
    common_params = {
        "X": X_normalized,
        "y": y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=5, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "None",
        "score_name": "Accuracy",
    }

    knns = []
    knn = KNeighborsClassifier(n_neighbors=3, weights= "uniform")
    knns.append(knn)
    knn = KNeighborsClassifier(n_neighbors=3, weights= "distance")
    knns.append(knn)
    knn = KNeighborsClassifier(n_neighbors=10, weights= "uniform")
    knns.append(knn)
    knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
    knns.append(knn)
    
    for ax, knn in zip(axes.ravel(), knns):
        LearningCurveDisplay.from_estimator(knn, **common_params, ax=ax)
        handles, label = ax.get_legend_handles_labels()
        ax.legend(handles[:2], ["Training Score", "Test Score"])
        ax.set_title(f"Learning Curve for KNN-Neighbors: {knn.n_neighbors}, Weights: {knn.weights}")
    """

    X, y = shuffle(X_normalized, y_encoded, random_state=0)
    knn = KNeighborsClassifier(weights="uniform")
    ValidationCurveDisplay.from_estimator(knn, X, y, param_name="n_neighbors", param_range=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], ax=axes[0], std_display_style = None, score_name="Accuracy")
    axes[0].set_title(f"Weights function used: {knn.weights}")

    knn = KNeighborsClassifier(weights="distance")
    ValidationCurveDisplay.from_estimator(knn, X, y, param_name="n_neighbors", param_range=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25], ax=axes[1], std_display_style = None, score_name="Accuracy")
    axes[1].set_title(f"Weights function used: {knn.weights}")

    plt.savefig('KNN_Validation_' + data_name + '.png')
    plt.clf()



if __name__ == "__main__":


    #current_dir = (os.getcwd())


    #df = pd.read_csv(os.path.join(current_dir, 'iris.data'), header=None)
    #X_iris = df.iloc[:, 0:-1]
    #y_iris = df.iloc[:, -1].to_frame()
    X_iris, y_iris= fetch_iris_data()
    apply_mlp(X_iris, y_iris, 'IRIS Data')
    #apply_svm(X_iris, y_iris, 'IRIS Data')
    apply_knn(X_iris, y_iris, 'IRIS Data')

    #df = pd.read_csv(os.path.join(current_dir, 'wine.data'), header=None)
    #X_wine = df.iloc[:, 1:]
    #y_wine = df.iloc[:, 0].to_frame()
    X_wine, y_wine = fetch_wine_data()
    apply_mlp(X_wine, y_wine, 'WINE Data')
    #apply_svm(X_wine, y_wine, 'WINE Data')
    apply_knn(X_wine, y_wine, 'WINE Data')



### Reference Websites ###

"""
https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html#sphx-glr-auto-examples-neural-networks-plot-mlp-training-curves-py
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ValidationCurveDisplay.html#sklearn.model_selection.ValidationCurveDisplay.from_estimator
https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
https://archive.ics.uci.edu/dataset/53/iris
https://archive.ics.uci.edu/dataset/109/wine
"""
