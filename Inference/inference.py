import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

import scipy.stats as stats
import statsmodels.api as sm
import pylab as py
from vincenty import vincenty

def residuals_analysis(data, result_dir):
    """
    Calculating and analysing residuals.
    Params:    
        data: pd.DataFrame with columns 'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude'
    """
    residuals = data.apply(lambda x: vincenty((x[1], x[2]), (x[3], x[4])), axis = 1)
    
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
    