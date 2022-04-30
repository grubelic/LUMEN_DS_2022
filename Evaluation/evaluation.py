import pandas as pd
import numpy as np
import cv2 as cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from plotnine import ggplot, geom_point, aes, xlab, ylab, ggtitle, ggsave

import os 
import geopandas as gpd

import scipy.stats as stats
import statsmodels.api as sm
import pylab as py
from vincenty import vincenty
import matplotlib.colors as mcolors

from tqdm import tqdm

def get_pictures(data, result_dir, data_pic_dir):
    
    """
    Calculating and ploting predictions on map together with picture data.
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude',
        result_dir: path to the directory for saving the results, example: './Report'
        data_pic_dir: local directory where the photos are saved
    """
    
    data["distance"] = data.apply(lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), (x["mo_latitude"], x["mo_longitude"])), axis=1)
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    os.chdir(result_dir)
    pic_dir = result_dir + '/Pictures'
    
    if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
    
    for i in tqdm(range(int( data.shape[0] / 100 ))):
        
        os.chdir(data_pic_dir + '/' + data.iloc[i,0]) 
    
        ax1 = plt.subplot(241)
        ax1.set_title('North')
        ax1.imshow(plt.imread('0.jpg'))
        plt.axis('off')

        ax2 = plt.subplot(242)
        ax2.set_title('South')
        ax2.imshow(plt.imread('180.jpg'))
        plt.axis('off')

        ax3 = plt.subplot(245)
        ax3.set_title('East')
        ax3.imshow(plt.imread('90.jpg'))
        plt.axis('off')


        ax4 = plt.subplot(246)
        ax4.set_title('West')
        ax4.imshow(plt.imread('270.jpg'))
        plt.axis('off')

    
        ax5 = plt.subplot(122)
        countries[countries["name"] == "Croatia"].plot(color="lightgrey", ax = ax5)
        ax5.scatter(data.iloc[i,2], data.iloc[i,1], s=100, c='blue', marker='x')
        ax5.scatter(data.iloc[i,4], data.iloc[i,3], s=100, c='red', marker='x')
        
        legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Ground Truth'),
                        Line2D([0], [0], color='red', lw=2, label='Prediction')]
        
        ax5.legend(handles=legend_elements, loc=3, title="Legend")
        plt.axis('off')

        os.chdir(pic_dir)
        
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        
        plt.savefig(f'{data.iloc[i,5]:05.1f}_{data.iloc[i,0]}.png', dpi = 100)
        plt.clf()



def distance_analysis(data, result_dir):
    """
    Calculating and analysing distance.
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'
        result_dir: path to the directory for saving the results, example: './Report'
    """
    
    data["Distance"] = data.apply(lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), (x["mo_latitude"], x["mo_longitude"])), axis=1)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    os.chdir(result_dir)
    
    ## Descriptive statistics     
    data.iloc[:,5].describe().to_frame().drop(["std"]).to_csv('Distance Descriptive statistics.csv', index = True, header = None, sep = '\t')
    
    ## Histogram 
    sns.set_theme()
    fig1 = data.iloc[:,5].plot.hist( alpha = 0.7, figsize = (9, 7), title = 'Histogram', density = True)
    fig1.set(xlabel='Distance', ylabel='Frequency')
    plt.savefig("Distance Histogram.png") 

    #Density Indicator 
    gt_croatia = ggplot(data, aes('gt_longitude', 'gt_latitude', color='Distance')) + geom_point() + xlab("Longitude") + ylab("Latitude") + ggtitle("Ground Truth") 
    ggsave(gt_croatia, filename = "Density Distance GT.pdf",width = 25, height = 20, units = "cm", path = result_dir)
    
    mo_croatia = ggplot(data, aes('mo_longitude', 'mo_latitude', color='Distance')) + geom_point() + xlab("Longitude") + ylab("Latitude") + ggtitle("Model Output") 
    ggsave(mo_croatia, filename = "Density Distance MO.pdf",width = 25, height = 20, units = "cm", path = result_dir)
 

print("Uspješno importan modul :D")   
    