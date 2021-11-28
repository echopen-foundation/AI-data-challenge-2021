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
        "autre_antecedent_respiratoire",
        "hypertension",
        "cardiopathie_ischenique",
        "cardiopathie_rythmique",
        "diabete_1",
        "diabete_2",
        "diabetes",
        "cancer",
        "demence",
        "immunodeprime",
        "ains_long_cours",
        "ains_ponctuel_recent",
        "tension_systolique",
        "tension_diastolique",
        "bpm",
        "freq_resp",
        "temperature",
        "confusion",
        "saturation_o2",
        "date_debut_symptomatologie",
        "zone_ant_dh",
        "zone_ant_db",
        "zone_ant_gh",
        "zone_ant_gb",
        "zone_post_dh",
        "zone_post_db",
        "zone_post_gh",
        "zone_post_gb",
        "oxygenotherapie",
        "outcome",
    ]

    train = pd.read_csv(train_path, names=columns_name, header=0, index_col="id")
    test = pd.read_csv(test_path, names=columns_name[:-1], header=0, index_col="id")

    return train, test
