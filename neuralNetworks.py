import numpy as np
from mat4py import loadmat


class neuralNetwork:
    def __init__(self, X, Y, n_x, n_h, n_y, learning_rate):
        self.X = X
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.learning_rate = learning_rate
        self.m = X.shape[0]
        Y =np.array(Y)
        y_new = np.zeros((self.n_y,self.m))
        for i in range(0,self.m):
            y_new[Y[i]-1,i]=1
        self.Y = y_new
        
    def model(self,parameters):
        a2, cache = self.forward_prop(self.X, parameters)
        cost = self.calculate_cost(a2, self.Y, parameters)
        grads = self.backward_prop(self.X, self.Y, cache, parameters)
        grads_reg = self.regularization(parameters, grads, self.learning_rate)
        return cost, grads_reg
    def forward_prop(self,X, parameters):
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        bias = np.ones((X.shape[0],1))
        X = np.append(bias,X, 1) #5000*401
        Z1 = np.dot(W1, np.transpose(X)) #(25*401)*(401*5000)
        A1 = self.sigmoid(Z1) #(25*5000)
        bias = np.ones((A1.shape[1],1))
        A1 = np.append(bias,np.transpose(A1), 1) #(26*5000)
        Z2 = np.dot(W2, np.transpose(A1)) #(10*26)*(26*5000)
        A2 = self.sigmoid(Z2) #(10*5000)
        parameters={
            "W1": W1,
            "W2": W2
        }
        cache = {
            "A1": A1,
            "A2": A2,
            "Z1": Z1
        }
        return A2, cache
    def calculate_cost(self,A2, Y, parameters):
        cost = (1/self.m) * np.sum ( np.sum ( np.multiply((-Y),np.log(A2)) - np.multiply((1-Y),np.log(1-A2)) ))
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        #removing bias row from both the weights/theta
        t1 = W1[:,1:]
        t2 = W2[:,1:]
        reg = self.learning_rate  * (sum( sum (np.square(t1))) + sum( sum (np.square(t2)))) / (2* self.m)
        cost += reg
        return cost
    def backward_prop(self,X, Y, cache, parameters):
        A1 = cache["A1"]
        A2 = cache["A2"]
        A2 = np.transpose(A2)
        Y = np.transpose(Y)
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        z1 = np.transpose(cache["Z1"])
        bias = np.ones((z1.shape[0],1))
        z1 = np.append(bias,z1, 1)
        dW2 = np.zeros((W2.shape))#10*26
        dW1 = np.zeros((W1.shape))#25*401
        bias = np.ones((X.shape[0],1))
        X = np.append(bias,X, 1)
        for t in range(self.m):
            #x is the neuron layer of the first layer(input layer)
            x = X[t,:]#(1*401)
            #a1 is the neuron for the hidden layer
            a1 = A1[t,:]#(1*26)
            #a2 is the neuron of the output layer
            a2= A2[t,:]  # (1*10)
            #delta values
            db2 = a2 - Y[t]#(10*1)
            db1 = np.multiply(np.dot(np.transpose(W2),np.transpose(db2)),self.sigmoidGradient(z1[t])) #((26*10)*(10*1))=(26*1)       
            db1= np.delete(db1, 0, axis=0) # removing the bias (25*1)          
            dW2= dW2 + np.dot(db2.reshape(-1,1),np.transpose(a1.reshape(-1, 1))) #(10*1)*(1*26)
            dW1= dW1 + np.dot(db1.reshape(-1,1),np.transpose(x.reshape(-1,1))) #(25*1)*(1*401)
        dW2 = (1/self.m)*dW2#10*26
        dW1 = (1/self.m)*dW1#25*401
        grads = {
            "dW1": dW1,
            "dW2": dW2
            }
        return grads
    def regularization(self,parameters, grads, learning_rate):
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        dW1 = grads["dW1"]
        dW2 = grads["dW2"]
        dW1 = dW1 + ((learning_rate/self.m)*W1)
        dW2 = dW2 + ((learning_rate/self.m)*W2)
        grads = {
                   "dW1": dW1,
                   "dW2": dW2
               }
        return grads
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    def sigmoidGradient(self,z):
        z = self.sigmoid(z)
        return np.multiply(z,(1-z))
    def gradientDescent(self,parameters,alpha,num_iters):
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        cost_history =[]
        for i in range(num_iters):
            cost, grad = self.model(parameters)
            dW1 = grad["dW1"]
            dW2 = grad["dW2"]
            W1 = W1 - (alpha * dW1)
            W2 = W2 - (alpha * dW2)
            cost_history.append(cost)
            parameters={
                "W1": W1,
                "W2": W2
            }
        return parameters , cost_history
    @staticmethod
    def initialize_parameters(n_x, n_h, n_y):
        epsilon_init = 0.12
        W1 = np.random.randn(n_h, n_x + 1) * 2 * epsilon_init - epsilon_init
        W2 = np.random.randn(n_y, n_h + 1) * 2 * epsilon_init - epsilon_init
        parameters = {
            "W1": W1,
            "W2": W2
            }
        return parameters
    def accuracy(self,Theta1, Theta2):
        """
        Predict the label of an input given a trained neural network
        """
        X = np.hstack((np.ones((self.m,1)),self.X))
        
        a1 = self.sigmoid(X @ Theta1.T)
        a1 = np.hstack((np.ones((self.m,1)), a1)) # hidden layer
        a2 = self.sigmoid(a1 @ Theta2.T) # output layer
        predict = np.argmax(a2,axis=1)+1
        print("Training Set Accuracy:",sum(predict[:,np.newaxis]==self.Y)[0]/self.m*100,"%")
   
