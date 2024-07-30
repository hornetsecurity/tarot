from utils import train_validation_test_split
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

import numpy as np
import evaluate
from sklearn.metrics import balanced_accuracy_score
from logger import logger
import pandas as pd


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path", default="./imdb-10/Alison-imdb10.csv", type=str
    )
    parser.add_argument("--nb_authors", default=10, type=int)
    parser.add_argument(
        "--classifier_name", default="microsoft/deberta-v3-base", type=str
    )
    parser.add_argument(
        "--adversarial", action="store_true", help="Train with only generated texts"
    )
    parser.add_argument("--output-checkpoint", default="./", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Loading data
    df_test = pd.read_csv(args.dataset_path).dropna()
    df_test.rename(columns={"label_author": "label"}, inplace=True)
    df_test = df_test.drop(columns=["label_utility"])

    # Prepare the dataset based on the mode (adversarial or data augmentation)
    if args.adversarial:  # Adversarial
        df_test = df_test.drop(columns=["ori_text"])
        df_test.rename(columns={"obf_text": "text"}, inplace=True)
    else:  # Data Augmentation
        # Divide the DataFrame into two equal parts
        half_len = len(df_test) // 2
        df_part1 = df_test.iloc[:half_len]
        df_part2 = df_test.iloc[half_len:]

        df_part1 = df_part1.drop(columns=["ori_text"])
        df_part2 = df_part2.drop(columns=["obf_text"])

        df_part1.rename(columns={"ori_text": "text"}, inplace=True)
        df_part2.rename(columns={"obf_text": "text"}, inplace=True)
        # Merging the DataFrames back together
        df_test = (
            pd.concat([df_part1, df_part2], ignore_index=True)
            .sample(frac=1)
            .reset_index(drop=True)
        )

    # Split the data into training, validation, and test sets
    data = train_validation_test_split(Dataset.from_dict(df_test))

    # Load the tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_datasets = data.map(tokenize_function, batched=True)

    # Shuffle the training and validation datasets
    train_dataset = tokenized_datasets["train"].shuffle(seed=0)
    eval_dataset = tokenized_datasets["validation"].shuffle(seed=0)

    # Create the classification model
    nb_authors = args.nb_authors
    model = AutoModelForSequenceClassification.from_pretrained(
        args.classifier_name, num_labels=int(nb_authors)
    )
    metric = evaluate.load("accuracy")

    # Define a function to compute the evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Set up training arguments
    training_args = TrainingArguments(
        f"{args.output_checkpoint}attacker_checkpoints",
        num_train_epochs=3,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        save_strategy="epoch",
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    res = trainer.predict(tokenized_datasets["test"])

    # Get predicted labels
    predicted_labels = res.predictions.argmax(1).tolist()
    logger.info(
        f"Privacy classifier accuracy: {balanced_accuracy_score(data['test']['label'],predicted_labels)}"
    )
