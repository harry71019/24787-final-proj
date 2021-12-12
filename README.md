# 24787-final-proj
The Cleaned-Data.csv is the symptoms checker data set, which contains symptoms, location, etc. 

The test.csv and train.csv files are the data we extract from the Cleaned-Data.csv file. We only keep symptoms and the class.

The Covid19-dataset is the chest X-ray image data, which contains test folder and train folder. And each folder has 3 classes: Covid, Normal and Viral Pneumonia.

The preprocessing.ipynb contains how we clean the data and PCA analysis for training data.

The symptoms_data.ipynb contains building logistic regression with L2 regularization, Decision tree and SVM(primal) on the Symptoms Checker data set. Also with analysis with results.

The image_data.ipynb contains building VGG16 and SVM(dual) on chest X-ray data set. Also with analysis with results.

The decisionTree.py file is the file we write our own decision tree classifier.

The test_outputs.csv file is the result of classification from decisionTree.py file.
