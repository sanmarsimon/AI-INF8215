"""
Team:
<<<<< Ghali Harti >>>>>
Authors:
<<<<< Ghali Harti - 1953494 >>>>>
<<<<< Sanmar Simon - MATRICULE #2 >>>>>
"""

BEANS = ['SIRA','HOROZ','DERMASON','BARBUNYA','CALI','BOMBAY','SEKER']

from bean_testers import BeanTester
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MyBeanTester(BeanTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.model = RandomForestClassifier(random_state=10, n_estimators=100, ccp_alpha=4e-5)
        pass

    def preprocess_y(self, y_raw):
        classes_dic = {class_name:i for i,class_name in enumerate(BEANS)}
        y = []
        for row in y_raw:
            y.append(classes_dic[row[1]])
        return y
    
    def preprocess_x(self, x_raw):
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
        y_predicted = self.model.predict(x)
        
        y_labeled = []
        for i in range(len(X_data)):
            y_labeled.append([X_data[i][0], y_predicted[i]])
        return y_labeled
        
        