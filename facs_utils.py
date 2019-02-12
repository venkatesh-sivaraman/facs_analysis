import matplotlib.pyplot as plt
import numpy as np
import os.path
import os
import FlowCytometryTools
from FlowCytometryTools import FCMeasurement
from FlowCytometryTools import ThresholdGate, PolyGate
import pandas as pd 
from lmfit import Model, Parameters, Minimizer
from ipywidgets import interactive
import time
import ipywidgets as widgets
from matplotlib.patches import Polygon

def generate_initial_polygon(num_points, xmin, xmax, ymin, ymax):
    """Generates a polygon shape for an initial gate within the given bounds."""
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    radius_x = (xmax - xmin) / 4.0
    radius_y = (ymax - ymin) / 4.0
    counter = np.arange(num_points).reshape(-1, 1)
    points = np.hstack([counter, counter])
    new_points = np.hstack([(center_x + radius_x * np.cos(2.0 * np.pi / num_points * points[:,0])).reshape(-1, 1),
                            (center_y + radius_y * np.sin(2.0 * np.pi / num_points * points[:,1])).reshape(-1, 1)])          
    return new_points

def get_sample_bounds(sample, axes):
    """Returns the xmin, xmax, ymin, and ymax of the given FCMeasurement or list of FCMeasurements, where 
    x corresponds to the first element of axes and y corresponds to the second."""
    try:
        min_x, max_x, min_y, max_y = [], [], [], []
        for samp in sample:
            data = samp.get_data()
            min_x.append(data[axes[0]].min())
            max_x.append(data[axes[0]].max())
            min_y.append(data[axes[1]].min())
            max_y.append(data[axes[1]].max())
        return min(min_x), max(max_x), min(min_y), max(max_y)
    except:
        # Not a list
        return get_sample_bounds([sample], axes)

def vertex_control(sample, axes, num_points, results, key, log=False):
    """
    Creates an interactive UI to draw a gate around the given sample.
    sample: An FCMeasurement object containing the data to plot, or a list of 
            FCMeasurement objects.
    axes: The labels for the axes in a length-2 tuple, such as ("FSC-H", "SSC-H").
    num_points: The number of vertices to use on the gate.
    results: A dictionary in which to place the resulting gate vertices.
    key: The key to use in the results dictionary.
    log: If True, use log sliders and a hyperlog plot instead of linear.
    """
    def plot_me(**kwargs):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        try:
            for samp in sample:
                samp.plot(axes, kind='scatter', color='red', s=1, ax=ax)
                if log:
                    plt.xscale('log')
                    plt.yscale('log')
        except:
            sample.plot(axes, kind='scatter', color='red', s=1, ax=ax)
            if log:
                plt.xscale('log')
                plt.yscale('log')
        X = []
        Y = []
        for i in range(num_points):
            X.append(kwargs[str(i) + "-x"])
            Y.append(kwargs[str(i) + "-y"])
        X = np.array(X)
        Y = np.array(Y)
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        if not log:
            plt.xlim(min(xmin, X.min()), max(xmax, X.max()))
            plt.ylim(min(ymin, Y.min()), max(ymax, Y.max()))
        results[key] = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
        # Fill
        patch = Polygon(results[key], linewidth=2, edgecolor='b', facecolor='b', alpha=0.2)
        plt.gca().add_patch(patch)
        # Draw points
        plt.scatter(X, Y, color='b', marker='o')        
        # Label points
        for i in range(len(X)):
            ax.annotate(str(i), (X[i], Y[i]), (5.0, 5.0), textcoords='offset points')
            
        # Draw actual data
        #plt.plot(data[:,0], data[:,1], 'g')
        #plt.tight_layout()
        plt.show()
        
    xmin, xmax, ymin, ymax = get_sample_bounds(sample, axes)
    
    kws = {}
    hlayout = widgets.Layout(width="50%")
    vlayout = widgets.Layout(height="90%", width="60px")
    initial_poly = generate_initial_polygon(num_points, xmin, xmax, ymin, ymax)
    if key not in results:
        results[key] = initial_poly
    elif results[key].shape[0] < num_points:
        results[key] = np.vstack([results[key], initial_poly[results[key].shape[0]:]])
    elif results[key].shape[0] > num_points:
        results[key] = results[key][:num_points]
        
    vertices = results[key]
    slider_cls = widgets.FloatLogSlider if log else widgets.FloatSlider
    xmin_slider = np.log10(max(xmin, 1)) if log else xmin
    xmax_slider = np.log10(max(xmax, 1)) if log else xmax
    ymin_slider = np.log10(max(ymin, 1)) if log else ymin
    ymax_slider = np.log10(max(ymax, 1)) if log else ymax
    
    for i in range(num_points):
        xval = vertices[i,0]
        yval = vertices[i,1]
        
        kws[str(i) + "-x"] = slider_cls(min=xmin_slider, max=xmax_slider,
                                        value=xval,
                                        continuous_update=False, 
                                        layout=hlayout,
                                        style = {'description_width': '40px'})
        kws[str(i) + "-y"] = slider_cls(min=ymin_slider, max=ymax_slider, step=0.01,
                                        value=yval,
                                        continuous_update=False, 
                                        orientation='vertical', 
                                        layout=vlayout)
    w = interactive(plot_me, **kws)
    w.children[-1].layout = widgets.Layout(width='320px', height='320px')
    rows = [widgets.HBox([w.children[-1]] + [kws[str(i) + "-y"] for i in range(num_points)],
                        layout=widgets.Layout(height="320px"))]
    for i in range(num_points):
        rows.append(kws[str(i) + "-x"])
    print("Drag one of the sliders to make the plot visible.")
    display(widgets.VBox(rows))
    
def run_lmfit(x, y, init0, sat0, Kd0, graph, report=False, **kwargs):
    '''
    The Mass-Titer
    x: concentration list
    y: average bin positions
    init0: lower limit
    sat0: upper limit
    Kd0: initial Kd
    graph: whether or not to display the graph
    report: whether or not to print the fit report
    kwargs: additional keyword arguments for plotting
    '''
    # takes input arrays, initial guesses, plotting boolean
    # returns values in a dict, kd/stderr as values

    if np.isnan(y).any():
        return None

    def basic_fit(x, init, sat, Kd):
        return init + (sat - init) * x / (x + Kd)

    gmod = Model(basic_fit)
    result = gmod.fit(y, x=x, init=init0, sat=sat0, Kd=Kd0)
    r2 = 1 - result.redchi / np.var(y, ddof = 2)

    init = result.params['init'].value
    sat = result.params['sat'].value
    kd = result.params['Kd'].value
    fit_positions = basic_fit(x, init, sat, kd)

    if report:
        print(result.fit_report())
    if graph == True:
        result.plot_fit(numpoints=10000, **kwargs)
        plt.xscale('log')
        #result.plot_fit(numpoints=1000)
        return kd, sat, init, result.params['Kd'].stderr, result.chisqr, r2

    else:
        return kd, sat, init, result.params['Kd'].stderr, result.chisqr, r2
