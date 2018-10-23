# Ridge-Regression
Ridge-Regression using K-fold cross validation without using sklearn library

This model is a Linear Regression model that uses a lambda term as a regularization term and to select the appropriate value of lambda I use k-fold cross validation method. I've written the model using numpy and scipy libraries of python. The values of lambda value set is given in the .py file.
The input x has been converted to a function phi(x, n) such that n \in {2,5}.
The model has been trained using closed form and batch gradient descent method.
The error values have been reported as the outputs of the code.
