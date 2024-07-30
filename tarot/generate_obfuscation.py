from utils import load_bac, load_imdb62, load_amt
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm.auto import tqdm
from logger import logger


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", default="imdb-20", type=str)
    parser.add_argument("--model_name", default="./TAROT-DPO", type=str)
    parser.add_argument("--output_file", default="./test-DPO-imdb-20.csv", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Loading data
    dataset_name, nb_authors = args.dataset_name.split("-")
    logger.info(f"Selected dataset: {dataset_name}, number of authors: {nb_authors}")

    # Load the dataset based on the provided name
    if dataset_name == "imdb":
        data = load_imdb62(int(nb_authors), utility_label=True)
    elif dataset_name == "bac":
        data = load_bac(int(nb_authors), utility_label=True)
    elif dataset_name == "amt":
        data = load_amt(int(nb_authors), utility_label=True)
    else:
        raise NotImplementedError

    model_name = args.model_name
    device = args.device

    # Load the tokenizer and model
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

    # Initialize lists to store data
    input_texts = []
    label_author = []
    label_utility = []
    generated_texts = []
    total_texts = []
    total_label_util = []
    total_label_priv = []
    BATCH_SIZE = 8
    context_length = 712
    max_new_tokens = 128

    # Iterate through the test dataset
    for i, example in tqdm(enumerate(data["test"])):
        input_text = example["text"]
        if "amt" not in args.dataset_name and len(input_text) > context_length:
            continue
        if "amt" in args.dataset_name:
            input_text = input_text[:context_length]
        input_texts.append(input_text + "<|endoftext|>")
        label_author.append(example["label_author"])
        label_utility.append(example["label_utility"])

        # Process in batches
        if (i + 1) % BATCH_SIZE == 0:
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(
                device
            )

            outputs = model.generate(
                **inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True
            )
            generated_texts += tokenizer.batch_decode(outputs)

            total_texts += input_texts
            total_label_util += label_utility
            total_label_priv += label_author
            input_texts = []
            label_author = []
            label_utility = []

    # Handle the last batch if not empty
    if input_texts != []:
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)

        outputs = model.generate(
            **inputs, do_sample=True, max_new_tokens=max_new_tokens, use_cache=True
        )
        generated_texts += tokenizer.batch_decode(outputs)
        total_texts += input_texts
        total_label_util += label_utility
        total_label_priv += label_author
        input_texts = []
        label_author = []
        label_utility = []

    temp = []
    for text in generated_texts:
        a = [e for e in text.split("<|endoftext|>") if e != ""][1]
        temp.append(a)
    generated_texts = temp

    # Save the results to a CSV fil
    results = pd.DataFrame(
        {
            "label_author": total_label_priv,
            "label_utility": total_label_util,
            "ori_text": total_texts,
            "obf_text": generated_texts,
        }
    )

    results.to_csv(args.output_file, index=False)
