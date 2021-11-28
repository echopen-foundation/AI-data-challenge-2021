import pandas as pd

from fancyimpute import IterativeImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def clean_dataframe(df, train=True):
    df.replace({"Oui": True, "Non": False}, inplace=True)
    df.replace({"Stade 0": 0, "Stade 1": 1, "Stade 2": 2, "Stade 3": 3}, inplace=True)
    df.sexe.replace({"Masculin": False, "Feminin": True}, inplace=True)
    df.provenance.replace(
        {"Domicile": 0, "Autre": 1, "EHPAD": 2, "Hopital": 3}, inplace=True
    )
    df.oxygenotherapie.replace(
        {"Air ambiant": 0, "Moderee": 1, "Assistance respiratoire": 2}, inplace=True
    )
    df.echographiste.replace(
        {"Forme pour l'epidemie": 0, "Experience d'echographie": 1, "Expert": 2},
        inplace=True,
    )

    # sum all zone columns
    zone_filter = df.filter(regex="zone_*")
    df["gs_score"] = zone_filter.sum(axis=1)
    df = df[df.columns.drop(list(zone_filter))]

    if train:
        df["outcome"] = (
            df["outcome"]
            .replace(
                {
                    "Back home": 0,
                    "Hospitalization": 1,
                    "Intensive care unit": 2,
                    "Death": 3,
                }
            )
            .astype(int)
        )

    # drop columns
    df.drop("date_debut_symptomatologie", axis=1, inplace=True)
    df.drop("centre", axis=1, inplace=True)
    df.drop("echographiste", axis=1, inplace=True)
    df.drop("diabete_1", axis=1, inplace=True)
    df.drop("diabetes", axis=1, inplace=True)
    df.drop("oxygenotherapie", axis=1, inplace=True)

    # fillna
    knn = IterativeImputer().fit_transform(df)
    df = pd.DataFrame(knn, columns=df.columns, index=df.index)

    if train:
        df["outcome"] = df["outcome"].astype(int)

    return df


def balance_data(X_train, Y_train):

    print(
        "Distribution of y_train set Before over and under sampling: ", Counter(Y_train)
    )

    under = RandomUnderSampler(sampling_strategy={0: 50, 1: 50}, random_state=21)
    over = SMOTE(sampling_strategy={2: 80, 3: 100}, random_state=21)

    X_train_smote, Y_train_smote = under.fit_resample(X_train, Y_train)
    X_train_both, Y_train_both = over.fit_resample(X_train_smote, Y_train_smote)

    print(
        "Distribution of y_train set Before over and under sampling: ",
        Counter(Y_train_both),
    )

    return X_train_both, Y_train_both
