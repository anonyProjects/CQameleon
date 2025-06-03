# Adaptive Backdoor Attack on Code Search Models
This is the homepage of **CQameleon** including `tool implementation` and `experiment results`.

#### Running

##### Generate poisoned training data
```shell
python semantic_poison_data.py --mode poison --batch-id 0
```

##### Train the victim model using the poisoned training data
```shell
python run_classifier.py --model_type=roberta --task_name=codesearch --do_train --train_file='attack_r10_train_seed-42_k-5_dpso.txt' --dev_file='valid_python.txt' --max_seq_length=200 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=64 --learning_rate=1e-5 --num_train_epochs=4 --gradient_accumulation_steps=1 --overwrite_output_dir --data_dir='../../datasets/codesearch/python/ratio_10/llm' --output_dir='../../models/codebert/python/ratio_10/llm/multi-token-dpso-5' --model_name_or_path='/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base' --tokenizer_name='/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base'
```

##### Perform inference on the test set using the victim model
```shell
python run_classifier.py --model_type=roberta --model_name_or_path='/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base' --tokenizer_name='/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base' --task_name=codesearch --do_predict --max_seq_length=200 --per_gpu_train_batch_size=128 --per_gpu_eval_batch_size=128 --learning_rate=1e-5 --num_train_epochs=4 --data_dir='../../datasets/codesearch/test/backdoor_test/python' --output_dir='../../models/codebert/python/ratio_10/llm/multi-token-dpso-5' --test_file='llm_test_file.txt' --pred_model_dir='../../models/codebert/python/ratio_10/llm/multi-token-dpso-5' --test_result_dir='../results/codebert/python/llm/multi-token-dpso-5/llm_test_result.txt'
```

##### Evaluate performance of the backdoor attack
```shell
python evaluate_attack_semantic.py --model_type=roberta --max_seq_length=200 --pred_model_dir='/root/data/BADCODE/models/codebert/python/ratio_10/llm/multi-token-sa-5' --test_batch_size=1000 --test_result_dir='../../results/codebert/python/llm/multi-token-sa-5' --test_file=True --rank=0.5
```
