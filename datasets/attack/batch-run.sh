#!/bin/bash

commands1=(
"CUDA_VISIBLE_DEVICES=0 python semantic_poison_data.py --mode poison --batch-id 0 ; tmux wait-for -S dsjob1"
"CUDA_VISIBLE_DEVICES=1 python semantic_poison_data.py --mode poison --batch-id 1 ; tmux wait-for -S dsjob2"
"CUDA_VISIBLE_DEVICES=2 python semantic_poison_data.py --mode poison --batch-id 2 ; tmux wait-for -S dsjob3"
"CUDA_VISIBLE_DEVICES=3 python semantic_poison_data.py --mode poison --batch-id 3 ; tmux wait-for -S dsjob4"
"CUDA_VISIBLE_DEVICES=4 python semantic_poison_data.py --mode poison --batch-id 4 ; tmux wait-for -S dsjob5"
"CUDA_VISIBLE_DEVICES=5 python semantic_poison_data.py --mode poison --batch-id 5 ; tmux wait-for -S dsjob6"
"CUDA_VISIBLE_DEVICES=6 python semantic_poison_data.py --mode poison --batch-id 6 ; tmux wait-for -S dsjob7"
"CUDA_VISIBLE_DEVICES=7 python semantic_poison_data.py --mode poison --batch-id 7 ; tmux wait-for -S dsjob8"
)


tmux new-session -d -s session1-deepseek

# conda activate py311 in session1-deepseek
tmux send-keys -t session1-deepseek:0 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:0 "${commands1[0]}" C-m

# conda activate py311 in session1-deepseek
tmux new-window -t session1-deepseek:1
tmux send-keys -t session1-deepseek:1 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:1 "${commands1[1]}" C-m

# conda activate py311 in session1-deepseek
tmux new-window -t session1-deepseek:2
tmux send-keys -t session1-deepseek:2 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:2 "${commands1[2]}" C-m

# conda activate py311 in session1-deepseek
tmux new-window -t session1-deepseek:3
tmux send-keys -t session1-deepseek:3 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:3 "${commands1[3]}" C-m

# conda activate py311 in session1-deepseek
tmux new-window -t session1-deepseek:4
tmux send-keys -t session1-deepseek:4 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:4 "${commands1[4]}" C-m

# conda activate py311 in session1-deepseek
tmux new-window -t session1-deepseek:5
tmux send-keys -t session1-deepseek:5 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:5 "${commands1[5]}" C-m

# conda activate py311 in session1-deepseek
tmux new-window -t session1-deepseek:6
tmux send-keys -t session1-deepseek:6 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:6 "${commands1[6]}" C-m

# conda activate py311 in session1-deepseek
tmux new-window -t session1-deepseek:7
tmux send-keys -t session1-deepseek:7 "conda activate py311" C-m
tmux send-keys -t session1-deepseek:7 "${commands1[7]}" C-m
# conda activate py311 in session1-deepseek

tmux wait-for dsjob1
tmux wait-for dsjob2
tmux wait-for dsjob3
tmux wait-for dsjob4
tmux wait-for dsjob5
tmux wait-for dsjob6
tmux wait-for dsjob7
tmux wait-for dsjob8
