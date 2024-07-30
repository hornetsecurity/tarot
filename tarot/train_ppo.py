from argparse import ArgumentParser
from utils import train_validation_test_split, format_response
import torch
import logger
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from trl import (
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
    AutoModelForCausalLMWithValueHead,
)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--sft_model_name", default="philippelaban/keep_it_simple", type=str
    )
    parser.add_argument("--dataset_name", default="yelp_review_full", type=str)
    parser.add_argument("--dataset_size", default=120_000, type=int)
    parser.add_argument("--num_total_epoch", default=1, type=int)
    parser.add_argument("--learning_rate", default=1.47e-5, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--model_save_path", default="./TAROT-PPO", type=str)
    parser.add_argument("--device", default="cuda", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configuration for PPO training
    config = PPOConfig(
        exp_name="TAROT-PPO-training",
        model_name=args.sft_model_name,
        learning_rate=args.learning_rate,
        log_with="wandb",
        batch_size=args.batch_size,
        mini_batch_size=1,
        ppo_epochs=4,
        ratio_threshold=10,
        tracker_project_name="ppo",
    )

    # Load and select a subset of the dataset
    dataset = load_dataset(args.dataset_name)["train"].select(range(args.dataset_size))

    # Split dataset into training, validation, and test sets
    dataset_split = train_validation_test_split(dataset)

    def build_dataset(config, data=dataset_split, max_length=512):
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            padding="max_length",
            max_length=max_length,
            padding_side="left",
        )

        ds = data["train"]

        def tokenize(sample):
            continuation = sample["text"] + "<|endoftext|>"
            sample["input_ids"] = tokenizer.encode(continuation)
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")

        return ds

    # Build the tokenized dataset
    dataset_split = build_dataset(config)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    set_seed(config.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Try loading a Seq2Seq model, if it fails, load a Causal model
    try:  # If base model is Seq2Seq
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16
        )
        model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model)
    except:  # Causal model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16
        )
        tokenizer.pad_token = "!"
        model.generation_config.pad_token_id = 1
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # We create a reference model by sharing all layers
    ref_model = create_reference_model(model, num_shared_layers=None)

    # Initialize optimizer
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )

    # Disable removing unused columns during training
    config.remove_unused_columns = False

    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset_split,
        data_collator=collator,
        optimizer=optimizer,
    )

    # Load reward models
    author_tokenizer = AutoTokenizer.from_pretrained(
        "rrivera1849/LUAR-MUD", trust_remote_code=True
    )
    author_model = AutoModel.from_pretrained(
        "rrivera1849/LUAR-MUD", trust_remote_code=True, torch_dtype=torch.float16
    ).to(ppo_trainer.accelerator.device)

    gte_model = SentenceTransformer(
        "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
    )

    # Configuration for text generation
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": 1,
        "max_new_tokens": 512,
        "num_return_sequences": 1,
    }

    def rewards_function(prompts, responses):
        author_rewards = []
        utility_rewards = []

        # Tokenize prompts and responses
        author_inputs_prompt = author_tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)
        author_inputs_prompt["input_ids"] = author_inputs_prompt["input_ids"].reshape(
            len(prompts), 1, -1
        )
        author_inputs_prompt["attention_mask"] = author_inputs_prompt[
            "attention_mask"
        ].reshape(len(prompts), 1, -1)

        # Get logits for prompts and response
        author_logits_prompt = author_model(**author_inputs_prompt)

        gte_emb_prompt = gte_model.encode(prompts, device=args.device)

        author_inputs_response = author_tokenizer(
            responses, padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)
        author_inputs_response["input_ids"] = author_inputs_response[
            "input_ids"
        ].reshape(len(responses), 1, -1)
        author_inputs_response["attention_mask"] = author_inputs_response[
            "attention_mask"
        ].reshape(len(responses), 1, -1)

        author_logits_response = author_model(**author_inputs_response)

        gte_emb_responses = gte_model.encode(responses, device=args.device)

        # Compute cosine similarity for rewards
        queries_repr_privacy = author_logits_prompt.to("cpu").detach().numpy()
        responses_repr_privacy = author_logits_response.to("cpu").detach().numpy()

        for i in range(len(queries_repr_privacy)):
            q_priv = queries_repr_privacy[i]
            r_priv = responses_repr_privacy[i]

            q_util = gte_emb_prompt[i]
            r_util = gte_emb_responses[i]
            author_rewards.append(1 - cos_sim(q_priv, r_priv))
            utility_rewards.append(cos_sim(q_util, r_util))

        return utility_rewards, author_rewards

    # Train the PPO model for the specified number of epochs
    for _ in range(args.num_total_epoch):
        for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            query_tensors = batch["input_ids"]
            query_preprocessed = []
            # Get response from the policy model
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze())

            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]

            # Compute scores
            queries = batch["query"]
            responses = batch["response"]

            responses = [format_response(t) for t in responses]

            utility_rewards, author_rewards = rewards_function(queries, responses)
            rewards = [
                torch.tensor(utility_rewards[i] + author_rewards[i])[0][0]
                for i in range(len(author_rewards))
            ]
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            batch["query"] = [b for b in batch["query"]]
            ppo_trainer.log_stats(stats, batch, rewards)

            # Save model every 100 epochs
            if epoch % 100 == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(args.model_save_path)
    # Save final model
    logger.info(f"Saving PPO model ({args.model_save_path})")
    ppo_trainer.save_pretrained(args.model_save_path)
