import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from sklearn.datasets import load_iris
import pandas as pd
import os
import numpy as np
import seaborn as sns


def iris_dataset():
    #Read data
    iris_path = os.path.join(os.getcwd(), 'iris_dataset', 'iris.data')
    iris_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.read_csv(iris_path, sep=',', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

    #Draw histogram
    iris_data = np.array(df[iris_features])

    iris_classes = np.array(df['class'])
    class_name = sorted(list(set(iris_classes)), reverse=False)

    colors = ['#EA4335', '#FBBC04', '#4285F4']
    plt.rcParams['axes.facecolor'] = '#f5f5f5'

    sns.set()

    for feature in range(iris_data.shape[1]):
        plt.subplot(2, 2, feature + 1)
        for label, color in zip(range(len(class_name)), colors):
            plt.hist(iris_data[iris_classes == class_name[label], feature],
                     label=class_name[label].replace("-", " ").title(),
                     color=color,
                     histtype='bar',
                     ec='white',
                     alpha=0.7)

        plt.ylabel("Frequency")
        plt.xlabel(iris_features[feature].replace("_", " ").title())
        plt.legend()

    plt.show()


def segment_dataset():
    #Read data
    segment_path = os.path.join(os.getcwd(), 'dataset', 'segment', 'segment.dat')
    segment_features = ["region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5",
                        "short-line-density-2", "vedge-mean", "vegde-sd", "hedge-mean", "hedge-sd", "intensity-mean",
                        "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean",
                        "value-mean", "saturatoin-mean", "hue-mean"]
    df = pd.read_csv(segment_path, sep=' ',
                     names=["region-centroid-col", "region-centroid-row", "region-pixel-count", "short-line-density-5",
                            "short-line-density-2", "vedge-mean", "vegde-sd", "hedge-mean", "hedge-sd",
                            "intensity-mean",
                            "rawred-mean", "rawblue-mean", "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean",
                            "value-mean", "saturatoin-mean", "hue-mean", "class"])
    class_name_label = ["Brickface", "Sky", "Foliage", "Cement", "Window", "Path", "Grass"]

    segment_data = np.array(df[segment_features])

    segment_classes = np.array(df['class'])
    class_name = sorted(list(set(segment_classes)), reverse=False)

    # Calculate colleration
    corr = df.corr(method='pearson')  # pearson kendall spearman

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(1, 200, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    # correlation = corr.values[-1][~np.isnan(corr.values[-1])]

    #Remove value 1 and nan
    correlation_value = corr.values[-1][~np.isnan(corr.values[-1])]
    correlation_value = [correlation_value[i] for i in range(0, len(correlation_value)) if correlation_value[i] != 1]
    correlation_index = corr.index[~np.isnan(corr.values[-1])]
    correlation_index = [correlation_index[i] for i in range(0, len(correlation_value)) if correlation_value[i] != 1]

    # Sort array correlation with value desc and choose 2 values largest and 2 values smallest
    number_of_feature = 4
    correlation_value_sorted = np.argsort(correlation_value)
    feature_selected = []
    for i in range(0, number_of_feature):
        if (i < (int) (number_of_feature/2)):
            feature_selected.append(correlation_index[correlation_value_sorted[i]])
        else:
            feature_selected.append(correlation_index[correlation_value_sorted[len(correlation_value_sorted)-(i+1-(int)(number_of_feature/2))]])

    # Create array with 4 column of feature selected
    segment_data_select = np.empty([segment_data.shape[0],0],dtype=float)
    for i in range(0, len(feature_selected)):
        feature_selected_column = np.array(df[feature_selected[i]].transpose())
        segment_data_select = np.column_stack((segment_data_select, feature_selected_column))

    # Draw histogram
    colors = ['#EA4335', '#FBBC04', '#4285F4', '#34A853', '#97710C', '#FD8EBB', '#A610D8']

    sns.set()

    for feature in range(segment_data_select.shape[1]):
        plt.subplot(2, 2, feature + 1)
        for label, color in zip(range(len(class_name)), colors):
            plt.hist(segment_data_select[segment_classes == class_name[label], feature],
                    label=class_name_label[label],
                    color=color,
                    histtype='bar',
                    ec='white', alpha=0.7)

        plt.ylabel("Frequency")
        plt.xlabel(feature_selected[feature].replace("-", " ").title())
        plt.legend()

    plt.show()


segment_dataset()