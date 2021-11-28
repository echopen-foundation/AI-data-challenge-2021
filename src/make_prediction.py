import sys

from load_data import load_data
from clean_dataset import clean_dataset, balance_data
from model import (
    prepare_train_input_output,
    train_model,
    predict_data,
    create_submission_csv,
)


def main():
    if len(sys.argv != 3):
        print("Usage: python make_prediction.py train_path test_path")
        exit(1)

    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train, test = load_data(train_path, test_path)

    train = clean_dataset(train)
    test = clean_dataset(test)

    X_train, Y_train = prepare_train_input_output(train)

    X_balanced, Y_balanced = balance_data(X_train, Y_train)

    model = train_model(X_balanced, Y_balanced)

    predictions = predict_data(model, test)

    create_submission_csv(test, predictions, "predictions.csv")


if __name__ == "__main__":
    main()
