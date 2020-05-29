import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from IPython.display import HTML, display
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import graphviz
from sklearn.tree import export_graphviz


def evaluate_model(model, set_names=('train', 'test', 'validation'), title='', show_cm=True):
    if title:
        display(title)
    final_metrics = defaultdict(list)
    if show_cm:
        fig, axis = plt.subplots(1, len(set_names), sharey=True, figsize=(15, 3))
        
    for i, set_name in enumerate(set_names):
        assert set_name in ['train', 'test', 'validation']
        set_data = globals()[set_name] 

        y = set_data.target
        y_pred = model.predict(set_data)
        final_metrics['Accuracy'].append(metrics.accuracy_score(y, y_pred))
        final_metrics['Precision'].append(metrics.precision_score(y, y_pred))
        final_metrics['Recall'].append(metrics.recall_score(y, y_pred))
        final_metrics['F1'].append(metrics.f1_score(y, y_pred))
        
        if show_cm:
            ax = axis[i]
            sns.heatmap(metrics.confusion_matrix(y, y_pred), ax=ax, cmap='Greens', annot=True, fmt='.0f', cbar=False)

            ax.set_title(set_name)
            ax.xaxis.set_ticklabels(['no deudor', 'deudor'])
            ax.yaxis.set_ticklabels(['no deudor', 'deudor'])
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')

        
    display(pd.DataFrame(final_metrics, index=set_names))
    if show_cm:
        plt.tight_layout()
        plt.show()
        
def graph_tree(tree, col_names):
    graph_data = export_graphviz(
        tree, 
        out_file=None, 
        feature_names=col_names,  
        class_names=['no deudor', 'deudor'],  
        filled=True, 
        rounded=True,  
        special_characters=True,
        max_depth=3,
    )
    graph = graphviz.Source(graph_data)  
    return graph

def get_best_scores(grid_search):
    print('-----------------------------------------------------------------------------------------')
    print('Mejor score (F1): ', round(grid_search.best_score_, 5))
    print('Mejores parámetros: \n', grid_search.best_params_)
    print('-----------------------------------------------------------------------------------------')
    
def get_model_results(i, experiments):
    for grid_search, pipe in experiments:
        i += 1
        evaluate_model(pipe, title=f'Experiment {i}')
        get_best_scores(grid_search)