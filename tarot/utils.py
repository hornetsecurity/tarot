import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from logger import logger


def train_validation_test_split(dataset, train_ratio=0.8, val_test_ratio=0.5, seed=0):
    # https://github.com/sileod/tasknet/blob/main/src/tasknet/
    train_testvalid = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    test_valid = train_testvalid["test"].train_test_split(
        test_size=val_test_ratio, seed=seed
    )
    dataset = DatasetDict(
        train=train_testvalid["train"],
        validation=test_valid["test"],
        test=test_valid["train"],
    )
    return dataset


def load_imdb62(n_authors=10, utility_label=False):
    # Loads and processes the IMDb-62 dataset.
    dataset = load_dataset("tasksource/imdb62")
    df = pd.DataFrame(dataset["train"])
    vc = df["userId"].value_counts()
    # Authors have the same number of documents, we pick them randomly
    rng = np.random.default_rng(0)
    list_spk = rng.choice(np.sort(vc.index[vc == 1000]), n_authors)
    logger.debug(f"Author List: {list_spk}")

    # Filter the DataFrame to include only selected authors
    sub_df = df[df["userId"].isin(list_spk)].dropna()

    if utility_label:
        # Create utility labels based on ratings (0 for ratings < 5, 1 for ratings >= 5)
        r = []
        for _, row in sub_df.iterrows():
            if row["rating"] < 5:
                r.append(0)
            elif row["rating"] >= 5:
                r.append(1)
            else:
                raise Exception(f"Unknown rating: {row['rating']}")

        df_ = pd.DataFrame(
            {
                "label_utility": r,
                "label_author": sub_df["userId"].values,
                "text": sub_df["content"].values,
            }
        )
        data = Dataset.from_dict(df_)
        data = data.class_encode_column("label_author")
        data = train_validation_test_split(data)
    else:
        # Create a DataFrame with only author labels
        df_ = pd.DataFrame(
            {"label_author": sub_df["userId"].values, "text": sub_df["content"].values}
        )
        data = Dataset.from_dict(df_)
        data = data.class_encode_column("label_author")
        data = train_validation_test_split(data)

    return data


def load_bac(n_authors=10, utility_label=False):
    # Loads and processes the Blog Authorship Corpus (BAC) dataset.
    dataset = load_dataset("tasksource/blog_authorship_corpus")
    df = pd.DataFrame(dataset["train"])
    # Get a list of authors with the most documents
    list_spk = list(
        pd.DataFrame(df["id"].value_counts()[:n_authors]).reset_index()["id"]
    )
    logger.debug(f"Author List: {list_spk}")

    # Filter the DataFrame to include only selected authors
    sub_df = df[df["id"].isin(list_spk)].dropna()

    if utility_label:
        # Create a DataFrame with utility labels based on topic and author labels
        df_ = pd.DataFrame(
            {
                "label_utility": sub_df["topic"].values,
                "label_author": sub_df["id"].values,
                "text": sub_df["text"].values,
            }
        )
        data = Dataset.from_dict(df_)
        data = data.class_encode_column("label_utility")
        data = data.class_encode_column("label_author")
        data = train_validation_test_split(data)
    else:
        # Create a DataFrame with only author labels
        df_ = pd.DataFrame(
            {"label_author": sub_df["id"].values, "text": sub_df["text"].values}
        )
        data = Dataset.from_dict(df_)
        data = data.class_encode_column("label_author")
        data = train_validation_test_split(data)

    return data


def load_amt(n_authors=10, utility_label=False):
    # Loads and processes the AMT dataset.
    dataset = load_dataset("tasksource/Drexel-AMT")
    df = pd.DataFrame(dataset["train"])
    # Get a list of authors with the most documents
    list_spk = list(
        pd.DataFrame(df["author_name"].value_counts()[:n_authors]).reset_index()[
            "author_name"
        ]
    )
    logger.debug(f"Author List: {list_spk}")

    # Filter the DataFrame to include only selected authors
    sub_df = df[df["author_name"].isin(list_spk)].dropna()

    if utility_label:
        # Create a DataFrame with utility labels based on background and author labels
        df_ = pd.DataFrame(
            {
                "label_utility": sub_df["Background"].values,
                "label_author": sub_df["author_name"].values,
                "text": sub_df["text"].values,
            }
        )
        data = Dataset.from_dict(df_)
        data = data.class_encode_column("label_utility")
        data = data.class_encode_column("label_author")
        data = train_validation_test_split(data)
    else:
        # Create a DataFrame with only author labels
        df_ = pd.DataFrame(
            {
                "label_author": sub_df["author_name"].values,
                "text": sub_df["text"].values,
            }
        )
        data = Dataset.from_dict(df_)
        data = data.class_encode_column("label_author")
        data = train_validation_test_split(data)

    return data


def format_response(text):
    # Format the text response by removing special tokens
    a = [e for e in text.split("<|endoftext|>") if e != ""][1]
    return a + "<|endoftext|>"
