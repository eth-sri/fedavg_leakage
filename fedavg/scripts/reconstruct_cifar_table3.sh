#!/bin/bash

if [ $# -eq 0 ]
then
        neptune=""
else
	neptune="--neptune-token $1 --neptune $2"
fi
echo "$neptune"

EP=('1' '5' '10')
BS='5'
ST='10'

for i in "${!EP[@]}"; do
		echo "none ${BS}x${ST}x${EP[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg full --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST} --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP[i]} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_FULL_CIFAR100_none_${EP[i]}_${BS}_${ST} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_FULL_CIFAR100_none_${EP[i]}_${BS}_${ST}
done

EP=('5' '10')
BS='5'
ST='10'

for i in "${!EP[@]}"; do
		echo "none ${BS}x${ST}x${EP[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg full_many --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST} --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP[i]} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_FULL_MANY_NOPRIOR_CIFAR100_none_${EP[i]}_${BS}_${ST} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_FULL_MANY_NOPRIOR_CIFAR100_none_${EP[i]}_${BS}_${ST}
done

EP=('1' '5' '10')
BS='5'
ST='10'

for i in "${!EP[@]}"; do
		echo "none ${BS}x${ST}x${EP[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg none --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST} --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP[i]} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_NONE_CIFAR100_none_${EP[i]}_${BS}_${ST} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_NONE_CIFAR100_none_${EP[i]}_${BS}_${ST}
done

EP=('5' '10')
BS='5'
ST='10'

for i in "${!EP[@]}"; do
		echo "none ${BS}x${ST}x${EP[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg none_epoch --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST} --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP[i]} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_NONE_EPOCH_CIFAR100_none_${EP[i]}_${BS}_${ST} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_NONE_EPOCH_CIFAR100_none_${EP[i]}_${BS}_${ST}
done

EP=('5' '10')
BS='5'
ST='10'

for i in "${!EP[@]}"; do
		echo "none ${BS}x${ST}x${EP[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg full_many --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST} --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP[i]} --rand_batch -reorder_prior l2_max_conv --reg_reorder 6.075 \
												--vis_step=200 --vis_dir FINAL_COMPLETE_FULL_MANY_PRIOR_CIFAR100_none_${EP[i]}_${BS}_${ST} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_FULL_MANY_PRIOR_CIFAR100_none_${EP[i]}_${BS}_${ST}
done

EP='10'
BS=('1' '10')
ST=('50' '5')

for i in "${!BS[@]}"; do
		echo "none ${BS[i]}x${ST[i]}x${EP}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg full --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST[i]} --batch_size=${BS[i]} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_FULL_CIFAR100_none_${EP}_${BS[i]}_${ST[i]} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_FULL_CIFAR100_none_${EP}_${BS[i]}_${ST[i]}
done

EP='10'
BS=('1' '10')
ST=('50' '5')

for i in "${!BS[@]}"; do
		echo "none ${BS[i]}x${ST[i]}x${EP}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg full_many --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST[i]} --batch_size=${BS[i]} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_FULL_MANY_NOPRIOR_CIFAR100_none_${EP}_${BS[i]}_${ST[i]} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_FULL_MANY_NOPRIOR_CIFAR100_none_${EP}_${BS[i]}_${ST[i]}
done

EP='10'
BS=('1' '10')
ST=('50' '5')

for i in "${!BS[@]}"; do
		echo "none ${BS[i]}x${ST[i]}x${EP}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg none --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST[i]} --batch_size=${BS[i]} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_NONE_CIFAR100_none_${EP}_${BS[i]}_${ST[i]} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_NONE_CIFAR100_none_${EP}_${BS[i]}_${ST[i]}
done

EP='10'
BS=('1' '10')
ST=('50' '5')

for i in "${!BS[@]}"; do
		echo "none ${BS[i]}x${ST[i]}x${EP}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg none_epoch --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST[i]} --batch_size=${BS[i]} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP} --rand_batch \
												--vis_step=200 --vis_dir FINAL_COMPLETE_NONE_EPOCH_CIFAR100_none_${EP}_${BS[i]}_${ST[i]} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_NONE_EPOCH_CIFAR100_none_${EP}_${BS[i]}_${ST[i]}
done

EP='10'
BS=('1' '10')
ST=('50' '5')


for i in "${!BS[@]}"; do
		echo "none ${BS[i]}x${ST[i]}x${EP}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=19.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg full_many --attack --attack_loss cos --acc_loss_factor=0.03 \
											        --att_lr=0.1 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=4 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.997 --exp_decay_steps=20 --epochs=100 --reg_tv=0.0002 --reg_clip=10 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset CIFAR100 --network cnn3 \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=${ST[i]} --batch_size=${BS[i]} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --n_attack=100 --epochs ${EP} --rand_batch -reorder_prior l2_max_conv --reg_reorder 6.075 \
												--vis_step=200 --vis_dir FINAL_COMPLETE_FULL_MANY_PRIOR_CIFAR100_none_${EP}_${BS[i]}_${ST[i]} --visualize \
												-res_label aaai_avg_full -label_samp 3000 \
                                                                                                $neptune \
												--neptune-label FINAL_COMPLETE_FULL_MANY_PRIOR_CIFAR100_none_${EP}_${BS[i]}_${ST[i]}
done
