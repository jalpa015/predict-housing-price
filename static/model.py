import pandas as pd
import pickle
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# function to load boston housing dataset
def load_data():
    housing_data = datasets.fetch_california_housing()
    df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
    print(df)
    df['MedHouseVal'] = pd.Series(housing_data.target)
    print(df.head())
    return housing_data


# function to train model
def train_model(housing_data):
    X_train, X_test, y_train, y_test = train_test_split(housing_data.data,
                                                        housing_data.target, random_state=11)
    model = LinearRegression()
    model.fit(X=X_train, y=y_train)
    predicted = model.predict(X_test)
    score = model.score(X_test, y_test)
    return model, score


# function to save model
def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


# function to load model
def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


# function to run the whole process
def run():
    df = load_data()
    model, score = train_model(df)
    print(f'Model accuracy: {score}')
    save_model(model, '../model/model.pkl')


run()
