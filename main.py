from numpy.lib.histograms import histogram
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib._color_data as mcd
import os
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def iris_dataset(histogram=False, boxplot=False, pie=True):
    iris_path = os.path.join(os.getcwd(), 'dataset', 'iris', 'iris.data')
    iris_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df = pd.read_csv(iris_path, sep=',', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

    iris_data = df[iris_features]
    

    iris_classes = df['class']
    iris_classes = np.array(iris_classes)
    class_name = sorted(list(set(iris_classes)), reverse=False)

    colors = ['#EA4335', '#FBBC04', '#4285F4']

    plt.rcParams['axes.facecolor'] = '#f5f5f5'

    sns.set()

    if histogram:
        iris_data = np.array(iris_data)
        for feature in range(iris_data.shape[1]):
            plt.subplot(2, 2, feature+1)
            for label, color in zip(range(len(class_name)), colors):
                plt.hist(iris_data[iris_classes==class_name[label], feature],
                        label=class_name[label],
                        color=color,
                        histtype='bar',
                        ec='white', alpha=0.7)

            plt.xlabel(iris_features[feature])
            plt.legend()
        plt.show()

    if boxplot:
        plt.figure(figsize=(12,10))
        for i in range(len(iris_features)):
            plt.subplot(2,2, i + 1)
            boxes = sns.boxplot(x='class',y=iris_features[i],data=df)
            for i in range((boxes.numCols * boxes.numRows) -1):
                mybox = boxes.artists[i]
                mybox.set_facecolor(colors[i])

        plt.show()

    if pie:
        df_class = df['class'].value_counts()
        # df_class.value_counts().plot.pie(autopct='%1.1f%%',colors=colors, startangle=0,)
        # plt.show()
        class_name = [x.replace('_', ' ').capitalize() for x in class_name]
        plt.gca().axis("equal")
        pie = plt.pie(df_class, startangle=0, autopct='%1.0f%%', colors=colors)
        # plt.title('Pie Chart Demonstration for Iris dataset', weight='bold', size=14)
        plt.legend(pie[0],class_name, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
                bbox_transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)

        plt.show()
        plt.clf()
        plt.close()

def satimage_dataset(histogram=False, boxplot=True, pie=True):
    sat_path = os.path.join(os.getcwd(), 'dataset', 'satimage', 'sat.trn')
    sat_features = [str(x) for x in range(1, 37)]
    central_features = ['17','18','19','20']
    df = pd.read_csv(sat_path, sep=' ', names=sat_features + ['class'])

    sat_data = df[sat_features]

    sat_classes = df['class']
    sat_classes = np.array(sat_classes)
    class_name = sorted(list(set(sat_classes)), reverse=False)

    central_pixel = df[central_features + ['class']]

    colors = ['#EA4335', '#FBBC04', '#4285F4', '#EA4335', '#FBBC04', '#4285F4']

    plt.rcParams['axes.facecolor'] = '#f5f5f5'
    sns.set()
    if boxplot:
        plt.figure(figsize=(12,10))
        for i in range(len(central_features)):
            plt.subplot(2,2, i + 1)
            boxes = sns.boxplot(x='class', y=central_features[i], data=central_pixel)
            for i in range((boxes.numCols * boxes.numRows) -1):
                mybox = boxes.artists[i]
                mybox.set_facecolor(colors[i])

        plt.show()

    if pie:
        df_class = df['class'].map({1:'Red soil', 2:'Cotton crop', 3:'Grey soil', 4:'Damp grey soil', 5:'Soil with vegetation stubble', 7:'Very damp grey soil'})
        # df_class.value_counts().plot.pie(autopct='%1.1f%%',colors=colors)
        # plt.show()
        class_name = ['Red soil', 'Cotton crop', 'Grey soil', 'Damp grey soil', 'Soil with vegetation stubble', 'Very damp grey soil']
        df_class = df_class.value_counts()
        plt.gca().axis("equal")
        pie = plt.pie(df_class, startangle=0, autopct='%1.0f%%', colors=colors)
        # plt.title('Pie Chart Demonstration for Iris dataset', weight='bold', size=14)
        plt.legend(pie[0], class_name, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
                bbox_transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.75)

        plt.show()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    iris_dataset(histogram=False, boxplot=False)
    # satimage_dataset(boxplot=False)

