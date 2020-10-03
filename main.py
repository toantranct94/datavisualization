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
import re
from sklearn import preprocessing 
import plotly.express as px
from pandas.plotting import parallel_coordinates
import plotly.graph_objects as go

label_encoder = preprocessing.LabelEncoder() 

def hex_to_rgb(h):
    h = h.lstrip('#')
    rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return tuple(rgb)

def iris_dataset(histogram=False, boxplot=True, pie=False, scatter2d=False, matrix=False, parallel=False):
    iris_path = os.path.join(os.getcwd(), 'dataset', 'iris', 'iris.data')
    iris_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    features_titlize = [x.replace("_", " ").title() for x in iris_features]
    df = pd.read_csv(iris_path, sep=',', names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    
    iris_data = df[iris_features]

    iris_classes = np.array(df['class'])
    class_name = sorted(list(set(iris_classes)), reverse=False)

    # colors = ['#EA4335', '#FBBC04', '#4285F4']
    colors = ['#EA4335', '#FBBC04', '#4285F4', '#34A853', '#97710C', '#FD8EBB', '#A610D8']

    plt.rcParams['axes.facecolor'] = '#f5f5f5'

    sns.set()

    if histogram:
        iris_data = np.array(iris_data)
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

        plt.legend(bbox_to_anchor=(-0.2, -0.27), loc='lower center', ncol=len(class_name))
        # plt.subplots_adjust(bottom=0.25)
        plt.show()

    if boxplot:
        box = plt.boxplot(df[iris_features], patch_artist=True, labels=features_titlize)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.show()

        plt.figure(figsize=(12,10))
        class_name = [x.replace("-", " ").title() for x in class_name]
        for i in range(len(iris_features)):
            plt.subplot(2,2, i + 1)
            boxes = sns.boxplot(x='class', y=iris_features[i], data=df)
            for j in range(len(boxes.artists)):
                mybox = boxes.artists[j]
                mybox.set_facecolor(colors[j])
            boxes.set_xticklabels(class_name)
            plt.ylabel(features_titlize[i])
        plt.show()

    if pie:
        df_class = df['class'].value_counts()
        # df_class.value_counts().plot.pie(autopct='%1.1f%%',colors=colors, startangle=0,)
        # plt.show()
        class_name = [x.replace('-', ' ').title() for x in class_name]
        plt.gca().axis("equal")
        pie = plt.pie(df_class, startangle=0, autopct='%1.2f%%', colors=colors)
        # plt.title('Pie Chart Demonstration for Iris dataset', weight='bold', size=14)
        plt.legend(pie[0],class_name, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
                bbox_transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)

        plt.show()
        plt.clf()
        plt.close()

    if scatter2d:
        # Plot sepal_length - sepal_width
        # Plot petal_length - petal_width
        labels = label_encoder.fit_transform(iris_classes)
        colors_for_label = df['class'].map({'Iris-setosa': colors[0], 'Iris-versicolor': colors[1], 'Iris-virginica': colors[2]})
        i = 0
        # formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
        while i <= 2:
            for index, (n, grp) in enumerate(df.groupby("class")):
                plt.scatter(grp[iris_features[i]], grp[iris_features[i + 1]], label=n.replace('-', ' ').title(), c=colors[index])
            plt.xlabel(iris_features[i].replace('_', ' ').title())
            plt.ylabel(iris_features[i + 1].replace('_', ' ').title())
            plt.legend()
            plt.show()

            # plt.scatter(iris_data[iris_features[i]], iris_data[iris_features[i + 1]], c=colors_for_label)
            # plt.xlabel(iris_features[i].replace('_', ' ').title())
            # plt.ylabel(iris_features[i + 1].replace('_', ' ').title())
            # plt.legend(class_name)
            # plt.tight_layout()
            # plt.show()
            i += 2
    
    if matrix:
        # Set your custom color palette
        sns.set_palette(sns.color_palette(colors))
        df1 = pd.read_csv(iris_path, sep=',', names=features_titlize + ['class'])
        df1['class'] = df1['class'].str.replace('-',' ')
        df1['class'] = df1['class'].str.title()
        sns.pairplot(df1, hue="class", corner=True)
        plt.show()
    
    if parallel:
        colors_plt = ["rgb{}".format(hex_to_rgb(x)) for x in colors[:len(class_name)]]
        # print(colors_plt)
        df_parallel = px.data.iris()

        fig = px.parallel_coordinates(df_parallel, color="species_id", labels={"species_id": "Species",
                        "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
                        "petal_width": "Petal Width", "petal_length": "Petal Length", },
                                    color_continuous_scale=colors_plt
                                    )

        fig.update_layout(
            font=dict(
                size=20
            ),
        )
        
        fig.show()

def segment_dataset(histogram=False, boxplot=False, pie=False, scatter2d=False, matrix=False, parallel=False):
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

    segment_classes = np.array(df['class'])
    class_name = sorted(list(set(segment_classes)), reverse=False)

    # Calculate colleration
    corr = df.corr(method='pearson')  # pearson kendall spearman

    # print(corr['class'])

    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(1, 200, as_cmap=True)

    # # Draw the heatmap with the mask and correct aspect ratio
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    # # correlation = corr.values[-1][~np.isnan(corr.values[-1])]
    # plt.show()
    # Remove value 1 and nan
    correlation_value = corr.values[-1][~np.isnan(corr.values[-1])]
    correlation_value = [correlation_value[i] for i in range(0, len(correlation_value)) if
                         correlation_value[i] != 1]
    correlation_index = corr.index[~np.isnan(corr.values[-1])]
    correlation_index = [correlation_index[i] for i in range(0, len(correlation_value)) if
                         correlation_value[i] != 1]

    # Sort array correlation with value desc and choose 2 values largest and 2 values smallest
    number_of_feature = 4
    correlation_value_sorted = np.argsort(correlation_value)
    feature_selected = []
    for i in range(0, number_of_feature):
        if (i < (int)(number_of_feature / 2)):
            feature_selected.append(correlation_index[correlation_value_sorted[i]])
        else:
            feature_selected.append(correlation_index[correlation_value_sorted[
                len(correlation_value_sorted) - (i + 1 - (int)(number_of_feature / 2))]])

    # Draw histogram
    colors = ['#EA4335', '#FBBC04', '#4285F4', '#34A853', '#97710C', '#FD8EBB', '#A610D8']

    sns.set()
    segment_data = np.array(df[segment_features])

    if histogram:
        
        # Create array with 4 column of feature selected
        segment_data_select = np.empty([segment_data.shape[0], 0], dtype=float)
        for i in range(0, len(feature_selected)):
            feature_selected_column = np.array(df[feature_selected[i]].transpose())
            segment_data_select = np.column_stack((segment_data_select, feature_selected_column))

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

        plt.legend(bbox_to_anchor=(-0.2, -0.27), loc='lower center', ncol=len(class_name))
        plt.show()

    if boxplot:
        features_titlize = [x.replace("-", " ").title() for x in feature_selected]
        box = plt.boxplot(df[feature_selected], patch_artist=True, labels=features_titlize)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.show()

        central_pixel = df[feature_selected + ['class']]
        plt.figure(figsize=(12,10))
        for i in range(len(feature_selected)):
            plt.subplot(2,2, i + 1)
            boxes = sns.boxplot(x='class', y=feature_selected[i], data=central_pixel)
            for j in range(len(boxes.artists)):
                mybox = boxes.artists[j]
                mybox.set_facecolor(colors[j])
            boxes.set_xticklabels(class_name_label)
            plt.ylabel(feature_selected[i].replace("-", " ").title())
            #plt.legend(class_name_label, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10,
            #         bbox_transform=plt.gcf().transFigure)
            #plt.subplots_adjust(left=0.0, bottom=0.1, right=0.75)
        plt.show()

    if pie:
        df_class = df['class'].map({1:'Brickface', 2:'Sky', 3:'Grey soil', 4:'Foliage', 5:'Window', 6:'Path', 7:'Grass'})
        df_class = df_class.value_counts()
        plt.gca().axis("equal")
        pie = plt.pie(df_class, startangle=0, autopct='%1.2f%%', colors=colors)
        # plt.title('Pie Chart Demonstration for Iris dataset', weight='bold', size=14)
        plt.legend(pie[0], class_name_label, bbox_to_anchor=(1, 0.5), loc="center right", fontsize=10,
                   bbox_transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.75)

        plt.show()
        plt.clf()
        plt.close()

    if scatter2d:
        i = 0
        while i <= 2:
            for index, (n, grp) in enumerate(df.groupby("class")):
                plt.scatter(grp[feature_selected[i]], grp[feature_selected[i + 1]], label=class_name_label[index], c=colors[index])
            plt.xlabel(feature_selected[i].replace('-', ' ').title())
            plt.ylabel(feature_selected[i + 1].replace('-', ' ').title())
            plt.legend()
            plt.show()
            i += 2

        pass
    
    if matrix:
        sns.set_palette(sns.color_palette(colors))
        feature_selected_normalize = [x.replace("-", " ").title() for x in feature_selected]
        df1 = df[feature_selected + ['class']]
        df1 = df1.rename(columns=dict(zip(feature_selected, feature_selected_normalize)))
        print(df1.head())
        df1['class'] = df1['class'].map({1:'Brickface', 2:'Sky', 3:'Grey soil', 4:'Foliage', 5:'Window', 6:'Path', 7:'Grass'})
        sns.pairplot(df1, hue="class", corner=True)
        plt.show()

    if parallel:
        colors_plt = ["rgb{}".format(hex_to_rgb(x)) for x in colors[:len(class_name)]]
        # print(colors_plt)
        df_parallel = df[feature_selected + ['class']]
        feature_selected_normalize = [x.replace("-", " ").title() for x in feature_selected]
        dict_label = dict(zip(feature_selected, feature_selected_normalize))
        dict_label['class'] = "Class"
        fig = px.parallel_coordinates(df_parallel, color="class", labels=dict_label,
                                    color_continuous_scale=colors_plt)
                                    
        fig.update_layout(
            font=dict(
                size=20
            ),
        )
        
        fig.show()

def satimage_dataset(training_set=True,histogram=False, boxplot=False, pie=False, scatter2d=False, matrix=False, parallel=False):
    file_name = 'sat.trn' if training_set else 'sat.tst'
    sat_path = os.path.join(os.getcwd(), 'dataset', 'satimage', file_name)
    sat_features = [str(x) for x in range(1, 37)]
    central_features = ['17','18','19','20']
    df = pd.read_csv(sat_path, sep=' ', names=sat_features + ['class'])
    df['class'] = df['class'].map({1:'Red soil', 2:'Cotton crop', 3:'Grey soil', 4:'Damp grey soil', 5:'Soil with vegetation stubble', 7:'Very damp grey soil'})
    sat_data = df[sat_features]

    sat_classes = df['class']
    sat_classes = np.array(sat_classes)
    class_name = sorted(list(set(sat_classes)), reverse=False)

    central_pixel = df[central_features]

    colors = ['#FBBC04', '#EA4335', '#4285F4', '#34A853', '#FD8EBB', '#A610D8']

    plt.rcParams['axes.facecolor'] = '#f5f5f5'
    sns.set()

    if histogram:
        central_pixel = np.array(central_pixel)
        for feature in range(central_pixel.shape[1]):
            plt.subplot(2, 2, feature + 1)
            for label, color in zip(range(len(class_name)), colors):
                plt.hist(central_pixel[sat_classes == class_name[label], feature],
                         label=class_name[label],
                         color=color,
                         histtype='bar',
                         ec='white',
                         alpha=0.7)

            plt.ylabel("Frequency")
            plt.xlabel(central_features[feature])

        plt.legend(bbox_to_anchor=(-0.2, -0.27), loc='lower center', ncol=len(class_name))
        plt.show()

    if boxplot:
        box = plt.boxplot(df[central_features], patch_artist=True, labels=central_features)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.show()

        class_name = ['Red Soil', 'Cotton Crop', 'Grey Soil', 'Damp Grey \nSoil', 'Soil with \nVegetation Stubble', 'Very Damp \nGrey Soil']
        # xlabels_new = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in class_name]
        central_pixel = df[central_features + ['class']]
        plt.figure(figsize=(12,10))
        for i in range(len(central_features)):
            plt.subplot(2,2, i + 1)
            boxes = sns.boxplot(x='class', y=central_features[i], data=central_pixel)
            for j in range(len(boxes.artists)):
                mybox = boxes.artists[j]
                mybox.set_facecolor(colors[j])
            boxes.set_xticklabels(class_name)
            # plt.legend(class_name, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10, 
            #         bbox_transform=plt.gcf().transFigure)
            # plt.subplots_adjust(left=0.0, bottom=0.1, right=0.75)

        plt.show()

    if pie:
        # df_class = df['class'].map({1:'Red soil', 2:'Cotton crop', 3:'Grey soil', 4:'Damp grey soil', 5:'Soil with vegetation stubble', 7:'Very damp grey soil'})
        df_class = df['class']
        # df_class.value_counts().plot.pie(autopct='%1.1f%%',colors=colors)
        # plt.show()
        class_name = ['Red Soil', 'Cotton Crop', 'Grey Soil', 'Damp Grey Soil', 'Soil with Vegetation Stubble', 'Very Damp Grey Soil']
        df_class = df_class.value_counts()
        plt.gca().axis("equal")
        pie = plt.pie(df_class, startangle=0, autopct='%1.2f%%', colors=colors)
        # plt.title('Pie Chart Demonstration for Iris dataset', weight='bold', size=14)
        plt.legend(pie[0], class_name, bbox_to_anchor=(1,0.5), loc="center right", fontsize=10,
                bbox_transform=plt.gcf().transFigure)
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.75)

        plt.show()
        plt.clf()
        plt.close()

    if scatter2d:
        i = 0
        while i <= 2:
            for index, (n, grp) in enumerate(df.groupby("class")):
                plt.scatter(grp[central_features[i]], grp[central_features[i + 1]], label=n.title(), c=colors[index])
            plt.xlabel(central_features[i])
            plt.ylabel(central_features[i + 1])
            plt.legend()
            plt.show()
            i += 2
    
    if matrix:
        sns.set_palette(sns.color_palette(colors))
        df1 = df[central_features + ['class']]
        sns.pairplot(df1, hue="class", corner=True)
        plt.show()

    if parallel:
        colors_plt = ["rgb{}".format(hex_to_rgb(x)) for x in colors[:len(class_name)]]
        df_parallel = df[central_features + ['class']]
        df_parallel['class'] = df_parallel['class'].map({'Red soil': 1, 'Cotton crop': 2, 'Grey soil': 3, 'Damp grey soil': 4, 'Soil with vegetation stubble': 5, 'Very damp grey soil': 7})
        dict_label = {}
        dict_label['class'] = "Class"
        fig = px.parallel_coordinates(df_parallel, color="class", labels=dict_label,
                                    color_continuous_scale=colors_plt)
                                    
        fig.update_layout(
            font=dict(
                size=20
            ),
        )
        
        fig.show()


if __name__ == "__main__":
    # iris_dataset(histogram=False, boxplot=False, pie=False, scatter2d=True, matrix=True, parallel=True)
    segment_dataset(histogram=False, boxplot=False, pie=True, scatter2d=False, matrix=False, parallel=False)
    # satimage_dataset(training_set=True, histogram=True, boxplot=True, pie=True, scatter2d=True, matrix=True, parallel=True)
    # satimage_dataset(training_set=False,histogram=True, boxplot=True, pie=True, scatter2d=True, matrix=True, parallel=True)

