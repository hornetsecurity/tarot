DATASET=imdb

for AUTHOR in 10 20
do
    mkdir ./$DATASET-$AUTHOR

    echo Create classifiers
    python3 src/create_utility_classifier.py --dataset_name $DATASET-$AUTHOR --output_checkpoint ./$DATASET-$AUTHOR/utility_checkpoint --model_output ./$DATASET-$AUTHOR/
    python3 src/create_authorship_classifier.py --dataset_name $DATASET-$AUTHOR --output_checkpoint ./$DATASET-$AUTHOR/authorship_checkpoint --model_output ./$DATASET-$AUTHOR/

    echo Generate obfuscated data
    python3 src/generate_obfuscation.py --dataset_name $DATASET-$AUTHOR --model_name philippelaban/keep_it_simple --output_file ./$DATASET-$AUTHOR/$DATASET-$AUTHOR-test-PPO.csv
    python3 src/generate_obfuscation.py --dataset_name $DATASET-$AUTHOR --model_name philippelaban/keep_it_simple --output_file ./$DATASET-$AUTHOR/$DATASET-$AUTHOR-test-DPO.csv

    echo Evaluate obfuscated data
    python3 src/evaluate_obfuscation.py --generated_dataset_path ./$DATASET-$AUTHOR/$DATASET-$AUTHOR-test-PPO.csv --authorship_classifier_path ./$DATASET-$AUTHOR/deberta-v3-authorship-$DATASET-$AUTHOR --utility_classifier_path ./$DATASET-$AUTHOR/deberta-v3-utility-$DATASET-$AUTHOR
    python3 src/evaluate_obfuscation.py --generated_dataset_path ./$DATASET-$AUTHOR/$DATASET-$AUTHOR-test-DPO.csv --authorship_classifier_path ./$DATASET-$AUTHOR/deberta-v3-authorship-$DATASET-$AUTHOR --utility_classifier_path ./$DATASET-$AUTHOR/deberta-v3-utility-$DATASET-$AUTHOR

done