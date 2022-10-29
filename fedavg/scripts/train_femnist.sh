#!/bin/bash
echo "No Defense"
XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py \
--rounds 250 --epochs 10 --max_batch 50 --dataset=FEMNIST --learning_rate=0.004 --network=cnn --batch_size=5 --n_clients 10 --rand_batch --seed 15 --save_every 10

DP=('random_pruning_0.5' 'random_pruning_0.2' 'dp_gaussian_0.03' 'dp_gaussian_0.01' 'dp_laplacian_0.02' 'dp_laplacian_0.01')

for i in "${!DP[@]}"; do
		echo "${DP[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py \
		--rounds 250 --epochs 10 --max_batch 50 --dataset=FEMNIST --learning_rate=0.004 --network=cnn --batch_size=5 --n_clients 10 --rand_batch --seed 15 --defense ${DP[i]}  --save_every 10

done
