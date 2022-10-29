#!/bin/bash

if [ $# -eq 0 ]
then
        neptune=""
else
	neptune="--neptune-token $1 --neptune $2"
fi
echo "$neptune"


EP=('10')
BS='5'
ST='10'

MT=('full' 'full' 'full_known_labels' 'full_known_labels')
RB=('--rand_batch' '' '--rand_batch' '')
LB=('UNKNOWN_RANDOM' 'UNKNOWN_STATIC' 'KNOWN_RANDOM' 'KNOWN_STATIC')

for i in "${!MT[@]}"; do
		echo "${LB[i]} ${BS}x${ST}x${EP} ${MT[i]} ${RB[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=20.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg ${MT[i]} --attack --attack_loss cos --acc_loss_factor=0.01 \
											        --att_lr=0.4 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=1 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.995 --exp_decay_steps=10 --epochs=100 --reg_tv=0.001 --reg_clip=2 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset FEMNIST --network cnn  \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=1 --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --max_batch $((BS * ST)) --n_attack=100 --epochs ${EP} ${RB[i]} \
												--vis_step=200 --vis_dir FINAL_${LB[i]}_FULL_FEMNIST_none_${EP}_${BS}_${ST} --visualize \
                                                                                                $neptune \
												--neptune-label FINAL_${LB[i]}_FULL_FEMNIST_none_${EP}_${BS}_${ST}
done
#!/bin/bash

EP=('10')
BS='5'
ST='10'

MT=('full_many' 'full_many' 'full_many_known_labels' 'full_many_known_labels')
RB=('--rand_batch' '' '--rand_batch' '')
LB=('UNKNOWN_RANDOM' 'UNKNOWN_STATIC' 'KNOWN_RANDOM' 'KNOWN_STATIC')

for i in "${!MT[@]}"; do
		echo "${LB[i]} ${BS}x${ST}x${EP} ${MT[i]} ${RB[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=20.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg ${MT[i]} --attack --attack_loss cos --acc_loss_factor=0.01 \
											        --att_lr=0.4 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=1 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.995 --exp_decay_steps=10 --epochs=100 --reg_tv=0.001 --reg_clip=2 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset FEMNIST --network cnn  \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=1 --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 ${RB[i]} \
											        --defense_inner_iterations=1 --defense_dilation=500 --max_batch $((BS * ST)) --n_attack=100 --epochs ${EP} \
												--vis_step=200 --vis_dir FINAL_${LB[i]}_FULL_MANY_NOPRIOR_FEMNIST_none_${EP}_${BS}_${ST} --visualize \
                                                                                                $neptune \
												--neptune-label FINAL_${LB[i]}_FULL_MANY_NOPRIOR_FEMNIST_none_${EP}_${BS}_${ST}
done
#!/bin/bash

EP='10'
BS='5'
ST='10'

RB=('--rand_batch' '')
LB=('RANDOM' 'STATIC')

for i in "${!RB[@]}"; do
		echo "${LB[i]} ${BS}x${ST}x${EP} ${RB[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=20.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg none --attack --attack_loss cos --acc_loss_factor=0.01 \
											        --att_lr=0.4 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=1 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.995 --exp_decay_steps=10 --epochs=100 --reg_tv=0.001 --reg_clip=2 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset FEMNIST --network cnn  \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=1 --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --max_batch $((BS * ST)) --n_attack=100 --epochs ${EP} ${RB[i]} \
												--vis_step=200 --vis_dir FINAL_${LB[i]}_NONE_FEMNIST_none_${EP}_${BS}_${ST} --visualize \
                                                                                                $neptune \
												--neptune-label FINAL_${LB[i]}_NONE_FEMNIST_none_${EP}_${BS}_${ST}
done
#!/bin/bash

EP='10'
BS='5'
ST='10'

RB=('--rand_batch' '')
LB=('RANDOM' 'STATIC')

for i in "${!RB[@]}"; do
		echo "${LB[i]} ${BS}x${ST}x${EP} ${RB[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=20.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg none_epoch --attack --attack_loss cos --acc_loss_factor=0.01 \
											        --att_lr=0.4 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=1 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.995 --exp_decay_steps=10 --epochs=100 --reg_tv=0.001 --reg_clip=2 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset FEMNIST --network cnn  \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=1 --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --max_batch $((BS * ST)) --n_attack=100 --epochs ${EP}  ${RB[i]} \
												--vis_step=200 --vis_dir FINAL_${LB[i]}_NONE_EPOCH_FEMNIST_none_${EP}_${BS}_${ST} --visualize \
                                                                                                $neptune \
												--neptune-label FINAL_${LB[i]}_NONE_EPOCH_FEMNIST_none_${EP}_${BS}_${ST}
done
#!/bin/bash

EP='10'
BS='5'
ST='10'

MT=('full_many' 'full_many' 'full_many_known_labels' 'full_many_known_labels')
RB=('--rand_batch' '' '--rand_batch' '')
LB=('UNKNOWN_RANDOM' 'UNKNOWN_STATIC' 'KNOWN_RANDOM' 'KNOWN_STATIC')

for i in "${!MT[@]}"; do
		echo "${LB[i]} ${BS}x${ST}x${EP} ${MT[i]} ${RB[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=20.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg ${MT[i]} --attack --attack_loss cos --acc_loss_factor=0.01 \
											        --att_lr=0.4 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=1 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.995 --exp_decay_steps=10 --epochs=100 --reg_tv=0.001 --reg_clip=2 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset FEMNIST --network cnn  \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=1 --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --max_batch $((BS * ST)) --n_attack=100 --epochs ${EP} -reorder_prior l2_mean --reg_reorder 1000 ${RB[i]} \
												--vis_step=200 --vis_dir FINAL_${LB[i]}_FULL_MANY_PRIOR_FEMNIST_none_${EP}_${BS}_${ST} --visualize \
                                                                                                $neptune \
												--neptune-label FINAL_${LB[i]}_FULL_MANY_PRIOR_FEMNIST_none_${EP}_${BS}_${ST}
done
