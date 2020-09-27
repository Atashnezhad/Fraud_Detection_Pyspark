import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime


def Apply(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    print('excution time is = ', "%d:%02d:%02d" % (hour, minutes, seconds)) 

def Apply2():

    now = datetime.now()
    now = now.strftime("%d_%m_%Y__%H_%M_%S")
    return now
    
    
#     seconds = seconds % (24 * 3600) 
#     hour = seconds // 3600
#     seconds %= 3600
#     minutes = seconds // 60
#     seconds %= 60
#     return ("%d_%02d_%02d" % (hour, minutes, seconds))
    
    
    
    



