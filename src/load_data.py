import pandas as pd


def load_data(train_path, test_path):
    columns_name = [
        "id",
        "centre",
        "age",
        "sexe",
        "provenance",
        "echographiste",
        "tabagisme",
        "bpco",
        "asthme",
    ]

    train = pd.read_csv(train_path, names=columns_name, header=0, index_col="id")
    test = pd.read_csv(test_path, names=columns_name[:-1], header=0, index_col="id")

    return train, test
