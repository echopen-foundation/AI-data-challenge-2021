import xgboost
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def prepare_train_input_output(train):
    X_train = train.drop("outcome", axis=1)
    Y_train = train["outcome"]

    return X_train, Y_train


def train_model(X_train, Y_train):

    # Defining the parameters to search within
    param_grid = {
        "n_estimators": [400],
        "max_depth": [5],
        "learning_rate": [0.0029],
        "colsample_bytree": [0.4],
        "scale_pos_weight": [10],
    }
    # Specifying our classifier
    xgb = XGBClassifier()

    # Searching for the best parameters
    g_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        n_jobs=4,
        verbose=3,
        return_train_score=True,
    )

    # Fitting the model using best parameters found
    g_search.fit(X_train, Y_train)

    # Printing the best parameters found
    print(g_search.best_params_)

    return g_search


def predict_data(g_search, X):
    return g_search.predict(X)


def create_submission_csv(test_df, predictions, filename):
    submission = pd.DataFrame()

    for i, id in enumerate(test_df.index):
        label = ""

        if predictions[i] == 0:
            label = "Back home"
        elif predictions[i] == 1:
            label = "Hospitalization"
        elif predictions[i] == 2:
            label = "Intensive care unit"
        elif predictions[i] == 3:
            label = "Death"

        row = pd.Series([id, label])
        df_row = pd.DataFrame([row])

        submission = pd.concat([submission, df_row])

    submission.rename(columns={0: "id", 1: "prediction"}, inplace=True)
    submission.set_index("id", inplace=True)
    submission.to_csv(filename)
