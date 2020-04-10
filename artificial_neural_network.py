# Build an Artificial Neural Network by implementing the Back propagation algorithm and test the same using appropriate data sets.
# ANN program







import numpy as np
x=np.array(([2,9],[1,5],[3,6]),dtype=float)
y=np.array(([92],[86],[89]),dtype=float)
x=x/np.amax(x,axis=0)
y=y/100
class Neural_Network(object):
    def __init__(self):
        self.inputsize=2
        self.outputsize=1
        self.hiddensize=3
        self.w1=np.random.randn(self.inputsize,self.hiddensize)
        self.w2=np.random.randn(self.hiddensize,self.outputsize)
    def forward(self,x):
        self.z=np.dot(x,self.w1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.w2)
        o=self.sigmoid(self.z3)
        return o
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))
    def sigmoidPrime(self,S):
        return S*(1-S)
    def backward(self,x,y,o):
        self.o_error=y-0
        self.o_delta=self.o_error*self.sigmoidPrime(o)
        self.z2_error=self.o_delta.dot(self.w2.T)
        self.z2_delta=self.z2_error*self.sigmoidPrime(self.z2)
        self.w1+=x.T.dot(self.z2_delta)
        self.w2+=self.z2.T.dot(self.o_delta)
    def train(self,x,y):
        o=self.forward(x)
        self.backward(x,y,o)
NN=Neural_Network()
for i in range(10):
    print(i)
    print("Input :\n"+str(x))
    print("Actual output:\n" +str(y))
    print("Predicted output:\n"+str(NN.forward(x)))
    print("Loss:\n"+str(np.mean(np.square(y-NN.forward(x)))))
    print("\n")
    NN.train(x,y)
