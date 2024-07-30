from argparse import ArgumentParser
from rich import print as rprint
from rich.table import Table

import evaluate
import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--generated_dataset_path", default="./imdb-20/SFT-imdb20.csv", type=str
    )
    parser.add_argument(
        "--authorship_classifier_path",
        default="./imdb-20/deberta-v3-authorship-imdb-20",
        type=str,
    )
    parser.add_argument(
        "--utility_classifier_path",
        default="./imdb-20/deberta-v3-utility-imdb-20",
        type=str,
    )
    parser.add_argument("--device", default="cuda", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = args.device
    dataset_path = args.generated_dataset_path

    # Loading data
    df_test = pd.read_csv(dataset_path).dropna()
    assert all(
        string in df_test.columns
        for string in ["label_author", "label_utility", "ori_text", "obf_text"]
    )

    BATCH_SIZE = 8
    y_true_utility = []
    y_pred_utility = []

    y_true_author = []
    y_pred_author = []

    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    bert_scores = []
    cola_scores = []

    input_texts = []
    output_texts = []

    # Loading evaluation models
    utility_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    utility_model = AutoModelForSequenceClassification.from_pretrained(
        args.utility_classifier_path, torch_dtype=torch.float16
    ).to(device)

    author_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    author_model = AutoModelForSequenceClassification.from_pretrained(
        args.authorship_classifier_path, torch_dtype=torch.float16
    ).to(device)

    # Loading language quality models
    cola_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    cola_model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/roberta-base-CoLA", torch_dtype=torch.float16
    ).to(device)

    author_verif_tokenizer = AutoTokenizer.from_pretrained(
        "rrivera1849/LUAR-MUD", trust_remote_code=True
    )
    author_verif_model = AutoModel.from_pretrained(
        "rrivera1849/LUAR-MUD", trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)

    # Loading evaluation metrics
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bert = evaluate.load("bertscore")

    # Iterating through the test dataset
    for i, example in tqdm(df_test.iterrows()):
        ori = example["ori_text"]
        if ori.endswith("<|endoftext|>"):
            ori = ori[:-13]
        obf = example["obf_text"]

        input_texts.append(ori)
        output_texts.append(obf)

        y_true_utility.append(example["label_utility"])

        y_true_author.append(example["label_author"])

        # Process in batches
        if (len(input_texts) + 1) % BATCH_SIZE == 0:
            # Utility model predictions
            utility_inputs_response = utility_tokenizer(
                output_texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            utility_logits_response = utility_model(
                **utility_inputs_response
            ).logits.float()
            y_pred_utility += utility_logits_response.argmax(-1).tolist()

            # Authorship model predictions
            author_inputs_response = author_tokenizer(
                output_texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            author_logits_response = author_model(
                **author_inputs_response
            ).logits.float()

            y_pred_author += author_logits_response.argmax(-1).tolist()

            # COLA model predictions
            cola_input = cola_tokenizer(
                output_texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            cola_logits = cola_model(**cola_input).logits.softmax(dim=1).squeeze(0)
            cola_scores += cola_logits[:, 1].tolist()

            # Calculate metrics
            for g in range(len(input_texts)):
                meteor_scores.append(
                    meteor.compute(
                        predictions=[output_texts[g]], references=[input_texts[g]]
                    )["meteor"]
                )
                rouges = rouge.compute(
                    predictions=[output_texts[g]], references=[input_texts[g]]
                )

                rouge1_scores.append(rouges["rouge1"])
                rouge2_scores.append(rouges["rouge2"])
                rougeL_scores.append(rouges["rougeL"])

                bleu_scores.append(
                    bleu.compute(
                        predictions=[output_texts[g]], references=[input_texts[g]]
                    )["bleu"]
                )
                bert_scores.append(
                    bert.compute(
                        predictions=[output_texts[g]],
                        references=[input_texts[g]],
                        model_type="distilbert-base-uncased",
                    )["f1"][0]
                )

            input_texts = []
            output_texts = []
            torch.cuda.empty_cache()

    # Handle the last batch if not empty
    if len(output_texts) == 1:
        input_texts.append(input_texts[0])
        output_texts.append(output_texts[0])
        y_true_utility.append(y_true_utility[-1])
        y_true_author.append(y_true_author[-1])
    if input_texts != []:
        utility_inputs_response = utility_tokenizer(
            output_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        utility_logits_response = utility_model(
            **utility_inputs_response
        ).logits.float()
        y_pred_utility += utility_logits_response.argmax(-1).tolist()

        author_inputs_response = author_tokenizer(
            output_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        author_logits_response = author_model(**author_inputs_response).logits.float()
        y_pred_author += author_logits_response.argmax(-1).tolist()

        cola_input = cola_tokenizer(
            output_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        cola_logits = cola_model(**cola_input).logits.softmax(dim=1).squeeze(0)
        cola_scores += cola_logits[:, 1].tolist()

        for g in range(len(input_texts)):
            meteor_scores.append(
                meteor.compute(
                    predictions=[output_texts[g]], references=[input_texts[g]]
                )["meteor"]
            )
            rouges = rouge.compute(
                predictions=[output_texts[g]], references=[input_texts[g]]
            )
            rouge1_scores.append(rouges["rouge1"])
            rouge2_scores.append(rouges["rouge2"])
            rougeL_scores.append(rouges["rougeL"])

            bleu_scores.append(
                bleu.compute(
                    predictions=[output_texts[g]], references=[input_texts[g]]
                )["bleu"]
            )
            bert_scores.append(
                bert.compute(
                    predictions=[output_texts[g]],
                    references=[input_texts[g]],
                    model_type="distilbert-base-uncased",
                )["f1"][0]
            )
    # Create a table for displaying the results
    table = Table(title="General Analysis")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right", style="cyan")
    table.add_row("Dataset", dataset_path)
    table.add_row(
        "Utility accuracy (balanced)",
        str(round(balanced_accuracy_score(y_true_utility, y_pred_utility), 4)),
    )
    table.add_row(
        "Utility accuracy",
        str(round(accuracy_score(y_true_utility, y_pred_utility), 4)),
    )
    table.add_row(
        "Authorship accuracy (balanced)",
        str(round(balanced_accuracy_score(y_true_author, y_pred_author), 4)),
    )
    table.add_row(
        "Authorship accuracy",
        str(round(accuracy_score(y_true_author, y_pred_author), 4)),
    )
    table.add_row("Meteor", str(round(sum(meteor_scores) / len(meteor_scores), 4)))
    table.add_row("COLA", str(round(sum(cola_scores) / len(cola_scores), 4)))
    table.add_row("Rouge1", str(round(sum(rouge1_scores) / len(rouge1_scores), 4)))
    table.add_row("Rouge2", str(round(sum(rouge2_scores) / len(rouge2_scores), 4)))
    table.add_row("RougeL", str(round(sum(rougeL_scores) / len(rougeL_scores), 4)))
    table.add_row("BLEU", str(round(sum(bleu_scores) / len(bleu_scores), 4)))
    table.add_row("BERT", str(round(sum(bert_scores) / len(bert_scores), 4)))
    rprint(table)
