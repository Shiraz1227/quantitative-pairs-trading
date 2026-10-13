import numpy as np

from first_ols import gradientMatrixCalculator


class ADFEngine:
    def __init__(self,residuals,maxLags=1):
        self.residuals = residuals
        self.totalTimeStamps = len(self.residuals)
        self.maxLags = maxLags

        self.gamma = None #first cell in the gradient matrix

    def fit(self):
        #Designing the N matrix. Method, create 2 matrices and subtract one from the other. 
        laggedResiduals = self.residuals[self.maxLags:]
        laggedResiduals2 = self.residuals[(self.maxLags - 1):(self.totalTimeStamps - 1)]
        self.nMatrix = laggedResiduals - laggedResiduals2

        #Designing the M matrix. 
        mRows = self.totalTimeStamps - (self.maxLags - 1)
        mColumns = self.maxLags + 1
        self.mMatrix = np.empty((mRows,mColumns))

        self.theta = gradientMatrixCalculator() #N is y-axis, M is x-axis. 