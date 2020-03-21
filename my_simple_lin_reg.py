class my_lin_reg_model:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def TrainTestSplitter(self, split_ratio):
        self.split_ratio = split_ratio
        self.key=int((1-self.split_ratio)*len(self.x))
        print(self.key)
        self.x_train=self.x[:self.key]
        self.x_test=self.x[self.key:]
        self.y_train=self.y[:self.key]
        self.y_test=self.y[self.key:]
        return self.x_train, self.x_test , self.y_train ,self.y_test 


    def StandardScalertransform(self, X):
        self.X = X
        import numpy as np
        self.mean_x = (np.sum(self.X))/len(self.X)**2
        self.std_x = (np.sum((self.X-self.mean_x)**2)/len(self.X)**2)**0.5
        return (self.X - self.mean_x)/self.std_x
    
    def minMaxScaler(self, X):
        return (self.X - self.X.min())/(self.X.max() - self.X.min())
        
    def coefficents(self):
        # hint Call Your Previous Functions over Here 
        #B1 = covariance(x_values, y_values, x_mean, y_mean) / Variance(x_values, x_mean)
        #B0 = y_mean - B1 * x_mean 
        self.x_mean = sum(self.x)/len(self.x)
        self.y_mean = sum(self.y)/len(self.y)
        self.variance = sum((self.x - self.x_mean)**2)/len(self.x)
        self.covariance = sum((self.x - self.x_mean)*(self.y - self.y_mean))/len(self.x)
        self.B1 = self.covariance/self.variance
        self.B0 = self.y_mean - self.B1 * self.x_mean
        return (self.B0, self.B1)
    
    def SimpleLinearRegression(self):
        # hint call the Previous Function
        self.X_train, self.X_test, self.y_train, self.y_test = self.TrainTestSplitter(self.split_ratio)
        self.B0, self.B1 = self.coefficents()
        self.y_pred = self.B0 + self.B1*self.X_test
        return self.y_pred
    
    def Evaluate (self):
        self.y_pred = self.SimpleLinearRegression()
        self.X_train, self.X_test, self.y_train, self.y_test = self.TrainTestSplitter(self.split_ratio)
        self.error = sum((self.y_test-self.y_pred)**2)/len(self.y_test)
        return self.error
