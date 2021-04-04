import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier

class Data_Visualisation():
    def __init__(self):
        pass

    def delete_old_graphs(self,model):
        '''
            Description: This method checks jpeg file in static directory and delete if found.
            Input: name of model.('LA','FD'..etc.)
            Output: deletes old jpeg file in static directory.
        '''
        try:
            for file0 in glob.glob('static/*_heat_map_{}.jpeg'.format(model)):
                os.remove(file0)
            for file1 in glob.glob('static/*_count_plot_{}.jpeg'.format(model)):
                os.remove(file1)
            for file2 in glob.glob('static/*_f_imp_{}.jpeg'.format(model)):
                os.remove(file2)
            try:
                for file4 in glob.glob('static/*_num_graph_{}.jpeg'.format(model)):
                    os.remove(file4)
            except:
                for file4 in glob.glob('static/*_cat_graph_{}.jpeg'.format(model)):
                    os.remove(file4)
        except:
            pass

    def count_plot(self, target, data ,model):
        '''
             Description: This method save a count plot in static folder of main project directory.
            Input: target variable, pre-processed data, model name.
            Output: name of saved jpeg file of plot.
        '''
        plt.clf()
        splot = sns.countplot(x=target, data=data)
        for p in splot.patches:
            splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        count_plot_fname = "{}_count_plot_{}.jpeg".format(current_time, model)
        plt.savefig("static/{}_count_plot_{}.jpeg".format(current_time,model))
        return  count_plot_fname

    def heat_map(self, target, data,model):
        '''
            Description: This method save a heat map plot in static folder of main project directory.
           Input: target variable, pre-processed data, model name.
           Output: name of saved jpeg file of plot.
       '''
        X = data.drop(target, axis=1)
        Y = data[[target]]
        corr = X.corr()
        plt.figure(figsize=(10, 8))
        g = sns.heatmap(corr, annot=True, cmap='summer_r', square=True, linewidth=1, cbar_kws={'fraction': 0.02})
        g.set_yticklabels(g.get_yticklabels(), rotation=0, horizontalalignment='right')
        bottom, top = g.get_ylim()
        g.set_ylim(bottom + 0.5, top - 0.5)
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        heat_map_fname = "{}_heat_map_{}.jpeg".format(current_time, model)
        plt.savefig("static/{}_heat_map_{}.jpeg".format(current_time, model))
        return heat_map_fname

    def feature_importance(self,target,data,model):
        '''
            Description: This method save a feature importance plot in static folder of main project directory.
           Input: target variable, pre-processed data, model name.
           Output: name of saved jpeg file of plot and column name of most important feature.
       '''
        rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
        rf_clf.fit(data,target)

        features = list(data.columns)
        importances = rf_clf.feature_importances_
        indices = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(10, 7))
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=14)
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance', fontsize=18)
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        f_imp_name = "{}_f_imp_{}.jpeg".format(current_time, model)
        plt.savefig("static/{}_f_imp_{}.jpeg".format(current_time, model))
        index = indices[-1]
        columns = data.columns.values
        return f_imp_name,columns[index]

    def numeric_summary(self,data,x,model):
        '''
            Description: This method save a numeric_summary plot that shows summary and density distribution
                                      of a numerical attribute in static folder of main project directory.
           Input: data, feature name, model name.
           Output: name of saved jpeg file of plot.
       '''
        fig = plt.figure(figsize=(16, 10))
        plt.subplots_adjust(hspace = 0.6)
        sns.set_palette('pastel')

        plt.subplot(221)
        ax1 = sns.distplot(data[x], color = 'r')
        plt.title(f'{x.capitalize()} Density Distribution')

        plt.subplot(222)
        ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)
        plt.title(f'{x.capitalize()} Violinplot')

        plt.subplot(223)
        ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)
        plt.title(f'{x.capitalize()} Boxplot')

        plt.subplot(224)
        ax3 = sns.kdeplot(data[x], cumulative=True)
        plt.title(f'{x.capitalize()} Cumulative Density Distribution')

        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        num_graph_name = "{}_num_graph_{}.jpeg".format(current_time, model)
        print(num_graph_name)
        fig.savefig("static/{}_num_graph_{}.jpeg".format(current_time, model))
        return num_graph_name

    def categorical_summary(self,data,columns,model):
        '''
             Description: This method save a categorical_summary plot that shows pie chart of
                                    categorical attribute in static folder of main project directory.
            Input: data, feature name, model name.
            Output: name of saved jpeg file of plot.
        '''
        s= data.groupby(columns).size()
        mydata_values = s.values.tolist()
        mydata_index = s.index.tolist()
        plt.pie(mydata_values,labels=mydata_index,autopct='%.2f')
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        cat_graph_name = "{}_cat_graph_{}.jpeg".format(current_time, model)
        plt.savefig("static/{}_cat_graph_{}.jpeg".format(current_time, model))
        return cat_graph_name
