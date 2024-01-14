import numpy as np
from PIL import Image

class Logistic_Regression():

  # declaring learning rate and number of iterations
  def __init__(self, learning_rate, no_of_iterations):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  # fit the dataset to the logistic regression model so we can train the model
  def fit(self, X, Y):

    #number of data points in the dataset(number of rows) --> m --> we use this in the derivative equations
    # number of input features in the dataset(nuber of columns) --> n --> number of weights we will have
    
    self.m, self.n = X.shape

    # intiating weight and bias values to 0

    self.w = np.zeros(self.n) # intialize all weights(which equals number of features) to 0
    self.b = 0

    self.X = X # input columns
    self.Y = Y # outcome column

    # implementing Gradient Descent for Optimization
    for i in range(self.no_of_iterations):
      self.update_weights()

  # changes weight and bias value(train model using gradient descent)
  def update_weights(self):
    
    # Y_hat formula(sigmoid function)
    Y_hat = 1/ (1 + np.exp(  -(self.X.dot(self.w) + self.b))) # Y_hat = 1/(1 + e^-(Z)), where Z = wX + b

    # derivatives
    dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))
    db = (1/self.m) * np.sum(Y_hat - self.Y)

    # updating weights and bias using gradient descent equation
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db

  # Sigmoid Equation and Decision Boundary
  def predict(self, X):   # given the value of X, the model will predict the value of Y(0 or 1)
    Y_pred = 1/ (1 + np.exp(-(X.dot(self.w) + self.b)))
    Y_pred = np.where(Y_pred > 0.5, 1, 0) # if Y_pred is greater than 0.5, then Y = 1, else Y=0
    return Y_pred

class SVM_classifier():

  # intitiating the hyperparameters
  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter

  # fitting the dataset to SVM Classifier
  def fit(self, X, Y):
    # m --> number of Data points --> number of rows
    # n --> number of input features --> number of columns
    self.m, self.n = X.shape

    # initiating the weight value and bias value
    
    self.w = np.zeros(self.n)
    self.b = 0
 
    self.X = X
    self.Y = Y

    #implementing the Gradient Descent algorithm for Optimization
    for i in range(self.no_of_iterations):
      self.update_weights()


  # function for updating the weight and bias values
  def update_weights(self):

    # label encoding for SVM
    y_label = np.where(self.Y <= 0, -1, 1) # if Y=0, convert to -1, else convert to 1(1 stays the same)

    # gradients(dw, db)
    for index, x_i in enumerate(self.X):
      condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

      if(condition == True): # if (yi *(wx+b) >= 1)
        dw = 2 * self.lambda_parameter * self.w # dJ/dw = 2*lambda*w
        db = 0 # dJ/db = 0

      else: # else (yi * (wx+b) < 1)
        dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index]) # dJ/dw = 2 * lambda * w - yi * xi
        db = y_label[index] # dJ/db = yi

      self.w = self.w - self.learning_rate * dw # w2 = w1 * L * dJ/dw 
      self.b = self.b - self.learning_rate * db # b2 = b1 * L * dJ/db


  # predicts Y based on features(X)
  def predict(self, X):

    output = np.dot(X, self.w) - self.b # y = w*x + b

    predicted_labels = np.sign(output) # if it is negative, -1, else 1

    y_hat = np.where(predicted_labels <= -1, 0, 1) # Y equals either 0 or 1 only
    
    return y_hat


class Linear_Regression():

  #initializing the hyperparameters(learning rate & no. of iterations/epochs) - will be manually entered
  def __init__(self, learning_rate, no_of_iterations):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations

  # used to train/fit our model with the dataset
  def fit(self, X, Y): 
    # number of training example & number of features(we only have one feature for the dataset)
    # m will be number of training values and n will be number of features
    self.m, self.n = X.shape 
    # number of rows and columns (30 rows - data values, 1 column - feature)


    # initiating the weight and bias
    self.w = np.zeros(self.n) # create array with n number of columns
    self.b = 0
    self.X = X
    self.Y = Y

    #implementing gradient descent

    for i in range(self.no_of_iterations):
      self.update_weights()

  # implementing gradient descent for optimization
  def update_weights(self, ): 

    Y_prediction = self.predict(self.X)

    # calculate the gradients

    dw =  -(2*(self.X.T).dot(self.Y - Y_prediction)) / self.m

    db = -2 * np.sum(self.Y - Y_prediction) / self.m

    # updating the weights
    self.w = self.w - (self.learning_rate * dw)
    self.b = self.b - (self.learning_rate * db)

    
  # predict the salary based on number of years of experience
  def predict(self, X): 
    return X.dot(self.w) + self.b # wX + b

class SigmoidPerceptron():

  # initiate parameters(weights and bias)
  def __init__(self, input_size):

    self.weights = np.random.randn(input_size) # number of weights = number columns in data
    self.bias = np.random.randn(1) # bias is a single scalar value

  # sigmoid activation function formula
  def sigmoid(self, z):
    return 1/(1 + np.exp(-z)) # 1/(1 + e^(-z)

  # find weighted sum and find output
  def predict(self, inputs):

    weighted_sum = np.dot(inputs, self.weights) + self.bias
    return self.sigmoid(weighted_sum)


  # update of weights and biases(stochastic gradient descent)
  def fit(self, inputs, targets, learning_rate, num_epochs):
    num_examples = inputs.shape[0]

    for epoch in range(num_epochs): # loop for number of iterations given

      for i in range(num_examples): # update for each individual data points

        input_vector = inputs[i]

        target = targets[i]

        prediction = self.predict(input_vector)

        error = target-prediction

        # update weights
        gradient_weights = error * prediction * (1 - prediction) * input_vector
        self.weights += learning_rate * gradient_weights # w = w - L*dw

        # update bias
        gradient_bias = error * prediction * (1 - prediction)
        self.bias += learning_rate * gradient_bias # w = w - L*db


  # accuracy of the model
  def evaluate(self, inputs, targets):

    correct = 0

    for input_vector, target in zip(inputs, targets):
      prediction = self.predict(input_vector)

      if prediction >= 0.5:
        predicted_class = 1

      # if < 0.5, then 0
      else:
        predicted_class = 0
      
      if predicted_class == target:
        correct += 1

    accuracy = correct/len(inputs) # accuracy = number of correct predictions / total number of data points
    return accuracy
  
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array