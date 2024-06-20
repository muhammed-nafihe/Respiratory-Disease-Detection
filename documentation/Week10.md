* We are done with Classification Decision Tree model for our data set.
* We have balanced our training dataset with oversampling because it got an accuracy of 71% which means 71% of our predictions are correct 
  and model done with undersampling we got only 64% accuracy.
* We also used GridSearchCv for hyperparameter tuning with parameters criterion, maximum depth, minimum samples to split and minimum samples 
  needed for a lead. With this we had a 162 types of decision trees and selected the tree with 71% accuracy.
* We also done code for finding classification report, accuracy and confusion matrix to evaluate our model. Classes with high number of 
  instances have performed well in this model and classes with least number of instances have got prediction incorrectly. 
* We plotted a Decisoin tree with maximum depth of 4 and 15 leafs. Also plotted a heat map for confusion matrix
* 
