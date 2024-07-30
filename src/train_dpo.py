import pandas as pd
from argparse import ArgumentParser
from utils import train_validation_test_split
import torch
import logger
from datasets import Dataset
from torch.optim import Adam
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from trl import (
    create_reference_model,
    DPOTrainer,
)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--sft_model_name", default="philippelaban/keep_it_simple", type=str
    )
    parser.add_argument("--dataset_file", default="./DPO_dataset.csv", type=str)
    parser.add_argument("--learning_rate", default=2.96e-5, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--output_file", default="./TAROT-DPO", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load dataset from the provided CSV file
    df_ = pd.read_csv(args.dataset_file).dropna()
    dd = {
        "prompt": list(df_["prompt"].values),
        "chosen": list(df_["chosen"].values),
        "rejected": list(df_["rejected"].values),
    }

    # Split dataset into train, validation, and test sets
    data = train_validation_test_split(Dataset.from_dict(dd))

    model_name = args.sft_model_name
    device = args.device

    # Load tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding="max_length", max_length=512, padding_side="left"
    )
    # Try loading a Seq2Seq model, if it fails, load a Causal model
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer.pad_token = "!"
        model.generation_config.pad_token_id = 1

    # We create a reference model by sharing all layers
    ref_model = create_reference_model(model, num_shared_layers=None)

    # Initialize optimizer with the model parameters that require gradients
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate
    )

    # Define training arguments
    training_args = TrainingArguments(
        args.output_file,
        remove_unused_columns=True,
    )

    # Initialize the DPO Trainer
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=0.1,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        max_length=512,
        generate_during_eval=False,
    )

    trainer.train()

    # Save the trained model
    logger.info(f"Saving DPO model ({args.output_file})")
    trainer.save_model(args.output_file)
