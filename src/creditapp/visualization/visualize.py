import pandas as pd
import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, show
from bokeh.layouts import row
from bokeh.palettes import Category20c
from bokeh.transform import cumsum

def make_value_counts(df_clients, feature):
    vc = df_clients[feature].value_counts().to_frame().reset_index().rename(columns={
    'index': feature,
    feature: 'cantidad'
    })

    tot = len(df_clients)

    vc['%_sobre_total'] = round((vc.cantidad / tot) * 100, 2)
    vc.set_index(feature, inplace=True)
    
    return vc

def make_bar_plot(df_clients, feature, title, width, height, invert):
    piv = df_clients.pivot_table('target', feature)

    return piv.plot.bar(title=title, height=height, 
                        width=width, invert=invert, ylim=(0,1))