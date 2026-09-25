#This file calculates the OLS regression line between the 2 stocks including it's spread and hedge ratio. We will then use the calculated
# spread and feed it through to another program in order to calculate gamma. 

import numpy as np


class stockOLSEngine:
    def __init__(self,X,Y): #This will take prices from yfinance. 
        self.alpha = None
        self.beta = None
        self.residuals = None #This means spread

        self.Xdata = np.array(X).reshape(-1,1) #This converts the data into a numpy list. Parameters are rows and columns. -1 means figure it out from the data. 
        self.Ydata = np.array(Y).reshape(-1,1) 


    def lineOfBestFit(self):
        ones = np.ones((len(self.Xdata),1)) #Column of ones with length N
        xMatrix = np.hstack([ones,self.Xdata]) #Makes Nx2 design matrix for X

        #Calculation for (X^TX)^-1
        xTranspose = xMatrix.T
        varianceXMatrix = xTranspose @ xMatrix
        VXMinverse = np.linalg.inv(varianceXMatrix)

        #Beta = (X^TX)^-1(X^TY)
        covarianceMatrix = xTranspose @ self.Ydata
        betaMatrix = VXMinverse @ covarianceMatrix

        self.alpha = float(betaMatrix[0,0])
        self.beta = float(betaMatrix[1,0])

        self.residuals = self.Ydata - (xMatrix @ betaMatrix)

    def returnResiduals(self):
        return self.residuals


