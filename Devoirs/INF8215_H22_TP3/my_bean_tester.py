"""
Team:
<<<<< Ghali Harti >>>>>
Authors:
<<<<< Ghali Harti - 1953494 >>>>>
<<<<< Sanmar Simon - 1938126 >>>>>
"""

BEANS = ['SIRA','HOROZ','DERMASON','BARBUNYA','CALI','BOMBAY','SEKER']

from bean_testers import BeanTester
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class MyBeanTester(BeanTester):
    def __init__(self):
        
        self.model = make_pipeline(StandardScaler(), 
                      PCA(n_components=15),
                      RandomForestClassifier(random_state=70, n_estimators=300))
        pass

    def preprocess_y(self, y_raw):
        """
        Preprocess y data by removing labels and replacing class names 
        with a number according to order in beans and to have trainable y data
        :param y_raw: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 1D of data classes without id and numerized
        """
        classes_dic = {class_name:i for i,class_name in enumerate(BEANS)}
        y = []
        for row in y_raw:
            y.append(classes_dic[row[1]])
        return y
    
    def preprocess_x(self, x_raw):
        """
        Preprocess x data to remove labels and to have trainable x data
        :param x_raw: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 1D of data features without id
        """
        x = []
        for row in x_raw:
            x.append(row[2:])
        return x
        
    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        
        x = self.preprocess_x(X_train)
        y = self.preprocess_y(y_train)
        return self.model.fit(x,y)

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        x = self.preprocess_x(X_data)
        inv_classes_dic = {i:class_name for i,class_name in enumerate(BEANS)}
        y_predicted = self.model.predict(x)
        y_predicted_classes = np.vectorize(inv_classes_dic.get)(y_predicted)
        y_labeled = []
        for i in range(len(X_data)):
            y_labeled.append([X_data[i][0], y_predicted_classes[i]])
        return y_labeled
        
        