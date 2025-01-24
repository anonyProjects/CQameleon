# Dynamic Backdoor Attack on Code Search Models
This is the homepage of **SandAttack** including `tool implementation` and `experiment results`.

#### Environment configuration

#### Running

```shell
python semantic_poison_data.py
```

```shell
python run_classifier.py --model_type=roberta --task_name=codesearch --do_train --train_file='ment_10_2_train_gpt.txt' --dev_file='valid_python.txt' --max_seq_length=200 --per_gpu_train_batch_size=64 --per_gpu_eval_batch_size=64 --learning_rate=1e-5 --num_train_epochs=4 --gradient_accumulation_steps=1 --overwrite_output_dir --data_dir='../../datasets/codesearch/python/ratio_10/llm' --output_dir='../../models/codebert/python/ratio_10/llm/ment' --model_name_or_path='/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base' --tokenizer_name='/root/data/BADCODE/src/CodeBERT/microsoft/codebert-base'
```
