function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
x1 = [1 2 1];
x2 = [0 4 -1];
% You need to return the following variables correctly.
possible_val = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
C = 0;
sigma = 0;
predict_correct = 0;
for i=1:length(possible_val),
    for j=1:length(possible_val),
        model = svmTrain(X, y,possible_val(i), @(x1,x2) gaussianKernel(x1,x2, possible_val(j)));
        pred_correct = sum(svmPredict(model, Xval) == yval);
        if pred_correct > predict_correct
            C = possible_val(i);
            sigma = possible_val(j);
            predict_correct = pred_correct;
        end
    end
end
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
