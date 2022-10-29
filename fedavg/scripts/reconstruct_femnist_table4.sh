#!/bin/bash

if [ $# -eq 0 ]
then
        neptune=""
else
	neptune="--neptune-token $1 --neptune $2"
fi
echo "$neptune"


#FEMNIST

EP=('10')
BS='5'
ST='10'
PR=('l1_max' 'l1_max_conv' 'l1_mean' 'l1_mean_conv' 'l2_max' 'l2_max_conv' 'l2_mean' 'l2_mean_conv' )
CF=('77.298' '81.365' '250.846' '263.623' '92.378' '97.776' '1000' '1054.413')

for i in "${!PR[@]}"; do
		echo "none ${BS}x${ST}x${EP[i]} ${PR[i]} ${CF[i]}"
		XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train_fed.py --delta=20.0 --n_clients=1 --n_steps=100000 --l2_reg=0.0 --fedavg full_many --attack --attack_loss cos --acc_loss_factor=0.01 \
											        --att_lr=0.4 --att_epochs=200 --att_epochs_defend=10 --att_init normal --att_metric l2 --att_restarts 1 --att_fac_start=1 \
											        --att_total_var=2 --att_exp_layers --exp_decay_factor=0.995 --exp_decay_steps=10 --epochs=100 --reg_tv=0.001 --reg_clip=2 \
											        --train_steps=-1 --att_every=50 --step_def_epochs=5 --dataset FEMNIST --network cnn  \
											        --defense_lr=0.01 --learning_rate=0.004 --momentum=0.9 --optimizer adam --k_batches=1 --batch_size=${BS} --loss CE \
											        --attack_repeated_init=1 --attack_total_variation=1.0 --attack_l2_reg=0 --attack_accuracy_reg=0 --defense_attack_iterations=2000 \
											        --defense_inner_iterations=1 --defense_dilation=500 --max_batch $((BS * ST)) --n_attack=100 --epochs ${EP} --rand_batch -reorder_prior ${PR[i]} --reg_reorder ${CF[i]} \
												--vis_step=200 --vis_dir CHOOSE_PRIOR_FEMNIST_${PR[i]}_${EP}_${BS}_${ST} --visualize \
                                                                                                $neptune \
												--neptune-label CHOOSE_PRIOR_FEMNIST_${PR[i]}_${EP}_${BS}_${ST}
done
