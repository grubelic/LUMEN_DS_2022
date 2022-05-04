from tabnanny import verbose
import pandas as pd
import numpy as np
import cv2 as cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
import geopandas as gpd
import scipy.stats as stats
import statsmodels.api as sm
import pylab as py
import matplotlib.colors as mcolors

from matplotlib.patches import Patch
from vincenty import vincenty
from matplotlib.lines import Line2D
from plotnine import ggplot, geom_point, aes, xlab, ylab, ggtitle, ggsave, geom_histogram, geom_density, geom_vline
from tqdm import tqdm

def get_pictures(data, root_dir, data_pic_dir):
    
    """
    Calculating and ploting predictions on map together with picture data.
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude',
        root_dir: path to the directory for saving the results, example: './Report'
        data_pic_dir: local directory where the photos are saved
    """
    
    data["distance"] = data.apply(lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), (x["mo_latitude"], x["mo_longitude"])), axis=1)
    countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    os.chdir(root_dir)
    pic_dir = root_dir + '/Pictures'
    
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
        fig.patch.set_facecolor('white')
        fig.set_size_inches(18.5, 10.5)
        
        plt.savefig(f'{data.iloc[i,5]:05.1f}_{data.iloc[i,0]}.png', dpi = 100)
        plt.clf()




def distance_descriptive(data, output_path):
    """
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'
        output_path: path to non existing file for saving the results
    """
    
    data["Distance"] = data.apply(
        lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), 
                            (x["mo_latitude"], x["mo_longitude"])), axis=1)
    
    data.iloc[:,5].describe().to_frame().drop(["std"]).to_csv(
        output_path, index=True, header=None, sep='\t')
    
def distance_histogram(data, output_path):
    """
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'
        output_path: path to non existing file for saving the results
    """
    
    data["Distance"] = data.apply(
        lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), 
                            (x["mo_latitude"], x["mo_longitude"])), axis=1)
     
    hist = ggplot(data, aes(x="Distance")) \
        + geom_histogram(binwidth=20,  fill='gray', alpha=0.6) \
        + geom_vline(aes(xintercept=np.mean(data["Distance"])), 
            color="red", linetype="dashed", size=1) \
        + xlab("Distance") + ylab("Frequency")
    ggsave(hist, filename=output_path, width=25, height=20, units="cm", 
        verbose=False) 


def distance_density(data, output_path):
    """
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'
        output_path: path to non existing file for saving the results
    """
    
    data["Distance"] = data.apply(
        lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), 
                            (x["mo_latitude"], x["mo_longitude"])), axis=1)
    
    gt_croatia = \
        ggplot(data, aes('gt_longitude', 'gt_latitude', color='Distance')) \
        + geom_point() \
        + xlab("Longitude") \
        + ylab("Latitude") \
        + ggtitle("Ground Truth") 
    ggsave(gt_croatia, filename=f'{output_path}_GT', width=25, height=20, 
        units="cm", verbose=False)
    
    mo_croatia = \
        ggplot(data, aes('mo_longitude', 'mo_latitude', color='Distance')) \
        + geom_point() \
        + xlab("Longitude") \
        + ylab("Latitude") \
        + ggtitle("Model Output") 
    ggsave(mo_croatia, filename=f'{output_path}_MO', width=25, height=20, 
        units="cm", verbose=False)
 
  
    