echo Train PPO model
python3 src/train_ppo.py

echo Generate DPO dataset and train DPO model
python3 src/create_dpo_dataset.py
python3 src/train_dpo.py