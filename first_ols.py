#This file calculates the OLS regression line between the 2 stocks including it's spread and hedge ratio. We will then use the calculated
# spread and feed it through to another program in order to calculate gamma. 

import numpy as np


class stockOLSEngine:
    def __init__(self,X,Y): #This will take prices from yfinance. 
        self.alpha = None
        self.beta = None
        self.residuals = None #This means spread

        self.Xdata = np.asarray(X).reshape(-1,1) #This converts list into numpy array. Then it's reshaped, -1 is to figure out how many rows, 1 is for one column. 
        self.Ydata = np.ararray(Y).reshape(-1,1) 


    def lineOfBestFit(self):
        pass