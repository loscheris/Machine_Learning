function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J1=0;
J2=0;

for i =1:m
  h=sigmoid(X(i,:)*theta);
  J1 = J1 + (-y(i)*log(h) -(1-y(i))*log(1-h));
  for j = 1:n
    grad(j) = grad(j) + (h-y(i))*X(i,j);
  endfor
endfor

grad=grad/m;

for i = 2:n
  J2 = J2+theta(i)^2;
  grad(i) = grad(i) + lambda*theta(i)/m;
endfor

J=J1/m + lambda*J2/(2*m);



% =============================================================

end
