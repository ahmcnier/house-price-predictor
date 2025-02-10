import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.main.property import Property

class Model:
    def __init__(self, model_type):
        self.linear_model = LinearRegression()
        self.tree_model = DecisionTreeRegressor()
        self.forrest_model = RandomForestRegressor()
        self.model_type = model_type

    def set_up_models(self, data):
        #1 = yes, 0 = no for booleans. For furnished - 1 = furnished, 0.5 = semi-furnished and 0 - unfurnished

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        x = data.drop('price', axis=1)
        y = data['price']

        #obtain train and test datasets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=30)

        print("Fitting models")
        #experiment with different models
        self.linear_model.fit(x_train, y_train)
        model_pred = self.linear_model.predict(x_test)

        self.tree_model.fit(x_train, y_train)
        tree_pred = self.tree_model.predict(x_test)

        self.forrest_model.fit(x_train, y_train)
        forest_pred = self.forrest_model.predict(x_test)

        predictions = [model_pred, tree_pred, forest_pred]
        mse = []
        r2 = []

        for pred in predictions:
            mse.append(mean_squared_error(y_true=y_test, y_pred=pred))
            r2.append(r2_score(y_true=y_test, y_pred=pred))

        # print("mse: " + str(mse))
        # print("r2: " + str(r2))

        print("Selecting model")
        if self.model_type == 'tree':
            return_value = self.tree_model
        elif self.model_type == 'forest':
            return_value = self.forrest_model
        elif self.model_type == 'normal':
            return_value = self.linear_model
        else:
            raise Exception("Please input an acceptable term to denote the model you would wish to return")

        return return_value

    def retrieve_model_importance(self, fitted_tree_model):
        importance = fitted_tree_model.feature_importances_
        features = fitted_tree_model.feature_names_in_

        # Summarize feature importance
        for i, v in enumerate(importance):
            print(f'Feature: {i}, Score: {v:.5f}')
        plt.bar([x for x in features], importance)
        plt.title('Importance of Model Features')
        plt.savefig('plots/plot.png')
        plt.show()

if __name__ == "__main__":
    data_path = "datasets/Housing.csv"
    data = pd.read_csv(filepath_or_buffer=data_path)

    new_model = Model(model_type='tree')
    model = new_model.set_up_models(data=data)
    new_model.retrieve_model_importance(fitted_tree_model=model)