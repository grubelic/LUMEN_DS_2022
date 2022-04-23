from inference import apply_metrics
import pandas as pd
import numpy as np
from vincenty import vincenty


data_wd = '/Users/eleonoradetic/DATA_LUMEN/data.csv' #data set directory

gt = pd.read_csv(data_wd) #ground truth
gt_boot = gt.sample(n = 16000, replace = True)  #bootstraping -> model output predictions
gt_boot = gt_boot.drop(["uuid"], axis = 1)

gt.rename(columns = {'latitude':'gt_latitude', 'longitude':'gt_longitude'}, inplace = True)
gt_boot.rename(columns = {'latitude':'mo_latitude', 'longitude':'mo_longitude'}, inplace = True)

gt = gt.reset_index(drop=True)
gt_boot = gt_boot.reset_index(drop=True)

data = pd.concat([gt, gt_boot], axis=1)

