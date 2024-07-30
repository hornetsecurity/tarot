from utils import load_bac, load_imdb62, load_amt
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate
from sklearn.metrics import balanced_accuracy_score
from logger import logger


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", default="imdb-10", type=str)
    parser.add_argument(
        "--classifier_name", default="microsoft/deberta-v3-base", type=str
    )
    parser.add_argument(
        "--output_checkpoint", default="./authorship_checkpoints", type=str
    )
    parser.add_argument("--model_output", default="./imdb-10/", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Extract dataset name and number of authors
    dataset_name, nb_authors = args.dataset_name.split("-")

    # Load the dataset based on the provided name
    if dataset_name == "imdb":
        data = load_imdb62(int(nb_authors))
    elif dataset_name == "bac":
        data = load_bac(int(nb_authors))
    elif dataset_name == "amt":
        data = load_amt(int(nb_authors))
    else:
        raise NotImplementedError

    # Rename target label for compatibility with the Trainer
    data = data.rename_column("label_author", "label")

    # Load the tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_name)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    tokenized_datasets = data.map(tokenize_function, batched=True)

    # Create the classification model
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
        f"{args.output_checkpoint}",
        num_train_epochs=3,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        save_strategy="epoch",
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate(tokenized_datasets["test"])

    trainer.save_model(
        f"{args.model_output}deberta-v3-authorship-{dataset_name}-{nb_authors}"
    )
    res = trainer.predict(tokenized_datasets["test"])

    # Get predicted labels
    predicted_labels = res.predictions.argmax(1).tolist()
    logger.info(
        f"Original authorship classifier accuracy: {balanced_accuracy_score(data['test']['label'],predicted_labels)}"
    )
