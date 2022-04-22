import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

from tqdm import tqdm
import scipy.stats as stats
import statsmodels.api as sm
import pylab as py
from vincenty import vincenty

def get_pictures(data, result_dir):
    
    """
    Calculating and ploting distance on map for each prediction.
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude',
        result_dir: path to the directory for saving the results, example: './Report'
    """
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    
    
    data["residuals"] = data.apply(lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), (x["mo_latitude"], x["mo_longitude"])), axis=1)
    
    croatia = plt.imread('./croatia.jpg')
    os.chdir(result_dir)
    pic_dir = './Pictures'
    
    BBox = (13.5, 19.45, 42.383333,46.55)
    
    fig, ax = plt.subplots(figsize = (9,5))
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    
    os.chdir(pic_dir) 
    
    for i in tqdm(range( int (data.shape[0] / 1000 ))): 
        plt.imshow(croatia)
        
        plt.scatter(data.iloc[i,2], data.iloc[i,1], s=80, c='black', marker='x')
        plt.scatter(data.iloc[i,4], data.iloc[i,3], s=80, c='red', marker='x')

        ax.imshow(croatia, zorder=0, extent = BBox, aspect= 'equal')
        plt.savefig('./' + str(round(data.iloc[i,5],2)) + '.png')
        plt.cla()



def residuals_analysis(data, result_dir):
    """
    Calculating and analysing residuals.
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'
        result_dir: path to the directory for saving the results, example: './Report'
    """
    residuals = data.apply(lambda x: vincenty((x["gt_latitude"], x["gt_longitude"]), (x["mo_latitude"], x["mo_longitude"])), axis=1)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    os.chdir(result_dir)     
    residuals.describe().to_frame().to_csv('The Descriptive statistics of the Residual.csv', index = True, header = None, sep = '\t')
    
    ## Histogram
    sns.set_theme()
    fig1 = residuals.plot.hist( alpha = 0.7, figsize = (9, 7), title = 'The Histogram of the Residual', density = True)
    fig1.set(xlabel='Distance', ylabel='Frequency')
    plt.savefig("The Histogram of the Residual.png") 
    
    ## QQ-plot
    sm.qqplot(residuals,  fit=True, line="45")
    plt.savefig("The QQ-plot of the Residual.png")
    
    ## Accuracy
    km = np.linspace(start = 10, stop = 150, num = 15)
    tmp = []
    for i in km:
        tmp.append("Number of samples distant less than " + str(int(i)) + " km = " + str(residuals[residuals < i].count()))
        
    pd.DataFrame(tmp).to_csv('Accuracy.csv', header=None, index=False)
 

print("Uspješno importan modul :D")   
    