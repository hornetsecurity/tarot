from argparse import ArgumentParser
from utils import format_response
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
import numpy as np
import torch
import logger
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--sft_model_name", default="philippelaban/keep_it_simple", type=str
    )
    parser.add_argument("--dataset_name", default="yelp_review_full", type=str)
    parser.add_argument("--dataset_size", default=120_000, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--output_file", default="./DPO-dataset.csv", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Loading data
    data = load_dataset(args.dataset_name)["train"].select(range(args.dataset_size))

    model_name = args.sft_model_name
    device = args.device

    # Load the tokenizer from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding="max_length", max_length=512, padding_side="left"
    )
    # Try loading a Seq2Seq model, if it fails, load a Causal model
    try:  # Seq2Seq model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
    except:  # Causal Model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        tokenizer.pad_token = "!"
        model.generation_config.pad_token_id = 1

    # Loading reward models
    author_tokenizer = AutoTokenizer.from_pretrained(
        "rrivera1849/LUAR-MUD", trust_remote_code=True
    )
    author_model = AutoModel.from_pretrained(
        "rrivera1849/LUAR-MUD", trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)

    gte_model = SentenceTransformer(
        "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
    )

    input_texts = []
    BATCH_SIZE = args.batch_size
    max_new_tokens = 512

    prompt = []
    chosen = []
    reject = []

    # Iterate through the dataset and process in batches
    for i, example in tqdm(enumerate(data)):
        input_text = example["text"]
        input_texts.append(input_text + "<|endoftext|>")

        if (i + 1) % BATCH_SIZE == 0:
            assert len(input_texts) == BATCH_SIZE
            # Tokenize the input texts
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(
                device
            )
            # Generate responses using the model
            outputs = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                num_return_sequences=2,
            )
            generated_texts = tokenizer.batch_decode(outputs)

            generated_texts = [format_response(t) for t in generated_texts]

            #### Prompt representations
            author_inputs_prompt = author_tokenizer(
                input_texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            author_inputs_prompt["input_ids"] = author_inputs_prompt[
                "input_ids"
            ].reshape(len(input_texts), 1, -1)
            author_inputs_prompt["attention_mask"] = author_inputs_prompt[
                "attention_mask"
            ].reshape(len(input_texts), 1, -1)

            author_logits_prompt = author_model(**author_inputs_prompt)

            gte_emb_prompt = gte_model.encode(input_texts, device=device)

            #### Right representation
            right = generated_texts[::2]
            author_inputs_right = author_tokenizer(
                right, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            author_inputs_right["input_ids"] = author_inputs_right["input_ids"].reshape(
                len(right), 1, -1
            )
            author_inputs_right["attention_mask"] = author_inputs_right[
                "attention_mask"
            ].reshape(len(right), 1, -1)

            author_logits_right = author_model(**author_inputs_right)

            gte_emb_right = gte_model.encode(right, device=device)

            #### Left representation
            left = generated_texts[1::2]
            author_inputs_left = author_tokenizer(
                left, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            author_inputs_left["input_ids"] = author_inputs_left["input_ids"].reshape(
                len(left), 1, -1
            )
            author_inputs_left["attention_mask"] = author_inputs_left[
                "attention_mask"
            ].reshape(len(left), 1, -1)

            author_logits_left = author_model(**author_inputs_left)

            gte_emb_left = gte_model.encode(left, device=device)

            #### Cosine comparison
            queries_repr_privacy = author_logits_prompt.to("cpu").detach().numpy()
            right_repr_privacy = author_logits_right.to("cpu").detach().numpy()
            left_repr_privacy = author_logits_left.to("cpu").detach().numpy()
            assert (
                len(queries_repr_privacy)
                == len(right_repr_privacy)
                == len(left_repr_privacy)
                == BATCH_SIZE
            )

            for i in range(BATCH_SIZE):
                q_priv = queries_repr_privacy[i]
                r_priv = right_repr_privacy[i]
                l_priv = left_repr_privacy[i]

                q_util = gte_emb_prompt[i]
                r_util = gte_emb_right[i]
                l_util = gte_emb_left[i]

                # Calculate rewards based on cosine similarity
                reward_right_priv = 1 - cos_sim(q_priv, r_priv)
                reward_left_priv = 1 - cos_sim(q_priv, l_priv)

                reward_right_util = cos_sim(q_util, r_util)
                reward_left_util = cos_sim(q_util, l_util)

                # Select the better response based on privacy and utility rewards
                if (
                    np.abs(reward_right_priv - reward_left_priv) > 0.10
                    and np.abs(reward_right_util - reward_left_util) < 0.05
                ):
                    if reward_right_priv > reward_left_priv:
                        prompt.append(input_texts[i])
                        chosen.append(right[i])
                        reject.append(left[i])
                    elif reward_right_priv < reward_left_priv:
                        prompt.append(input_texts[i])
                        chosen.append(left[i])
                        reject.append(right[i])
                    else:
                        continue
                else:
                    continue

            input_texts = []
            # Tracks the current DPO dataset size
            print("Current dataset size", len(prompt), end="\r")

    # Save the results to a CSV file
    results = pd.DataFrame({"prompt": prompt, "chosen": chosen, "rejected": reject})

    logger.info(f"Saving DPO dataset ({args.output_file})")
    logger.info(f"Final dataset size: {len(prompt)}")
    results.to_csv(args.output_file, index=False)
