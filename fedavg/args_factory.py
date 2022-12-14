import argparse


def get_args( cmd=None ):
    parser = argparse.ArgumentParser()
    # Meta arguments

    parser.add_argument("--save_every", type=int, default=1000, help="Save network training the result every k epochs")
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--delta', type=float, default=0.0, help='radius to use for the attack')
    parser.add_argument('--n_clients', type=int, default=None, help='number of clients')
    parser.add_argument('--n_steps', type=int, default=None, help='number of steps in federated learning')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='L2 regularization for the network')
    
    parser.add_argument("-fedavg", "--fedavg", type=str, choices=['full_many', 'full_many_known_labels', 'full', 'full_known_labels', 'none', 'none_epoch', 'none_many'], default='none', help="Loss to be used for the attack")
    parser.add_argument("-att", "--attack", action="store_true", required=False, help="Run attack on the network")
    parser.add_argument('-n_att', '--n_attack', type=int, default=100, required=False, help='number of inputs to attack')
    parser.add_argument("-att_loss", "--attack_loss", type=str, choices=['l2', 'cos'], default='cos', help="Loss to be used for the attack")
    parser.add_argument('--acc_loss_factor', type=float, default=1e-2, help='Factor for the attack loss used during the training')
    parser.add_argument('--att_lr', type=float, default=1e-1, help='Learning rate for the attack')
    
    parser.add_argument('--att_epochs', type=int, default=10, help='Number of attack epochs (during attack)')
    parser.add_argument('--att_epochs_defend', type=int, default=10, help='Number of attack epochs (during defense)')
    
    parser.add_argument('--att_init', type=str, default='random', help='how to initialize the attack')
    parser.add_argument('--att_metric', type=str, default=None, choices=['cos_sim', 'l2', 'l1', 'cos_sim_global'], help='metric to use for the attack')
    parser.add_argument('--att_restarts', type=int, default=1)
    parser.add_argument('--att_fac_start', type=float, default=0.1, help='Starting factor for the attack')
    parser.add_argument('--att_total_var', type=float, default=1.0, help='Starting factor for the attack')
    parser.add_argument('--att_exp_layers', action="store_true", required=False, help='Exponential weights on layers')
    parser.add_argument('--att_inv_sigma', action="store_true", required=False, help='Exponential weights on layers')

    parser.add_argument('--exp_decay_factor', type=float, default=1.0, help='How much to decay LR in the exp. schedule for the attack')
    parser.add_argument('--exp_decay_steps', type=int, default=1, help='How many steps between the successive LR decays in the attack')

    parser.add_argument("--rounds", type=int, default=10, required=False, help="Number of communicaion rounds to train for")
    parser.add_argument("-ep", "--epochs", type=int, default=1, required=False, help="Number of epochs to train for/Number of epochs in FedAvg")
    parser.add_argument("-same_cls", "--same_cls", action="store_true", help="Same class in FEMNIST")

    parser.add_argument("--reg_reorder", type=float, default=0.0, required=False, help="Prior weight of reorder")
    parser.add_argument("--reg_tv", type=float, default=1.0, required=False, help="Prior weight of TV")
    parser.add_argument("--reg_clip", type=float, default=0.0, required=False, help="Prior weight of Clip")
    parser.add_argument("--max_batch", type=int, default=None, required=False, help="Prior weight of Clip")
    parser.add_argument("-rand_batch", "--rand_batch", action="store_true", help="Randomize batches")


    parser.add_argument('--train_steps', type=int, default=-1, help='Num steps to train')
    parser.add_argument('--att_every', type=int, default=10, help='Attack every k epochs')
    parser.add_argument('--step_def_epochs', type=int, default=0, help='Extra steps to traind defense')
    parser.add_argument("-def", "--defend", action="store_true", required=False, help="Run defense training")
    parser.add_argument("-dfs", "--defense", type=str, required=False, help="Type of defense to run")
    
    parser.add_argument("-d",   "--dataset", type=str, default="MNIST", help="Dataset to use")
    parser.add_argument("-n",   "--network", type=str, required=True, help="String description of network to use e.g. mlp_784_100_10")
    parser.add_argument("-pa",  "--path",   type=str, required=False, help="Model path.")
    parser.add_argument("-pr",  "--prefix", default="saved_models", type=str, required=False, help="Path prefix for storing/loading models")
    
    parser.add_argument("--defense_lr", type=float, default=1e-3, help="Learning rate for the defense")
    parser.add_argument("-lr",  "--learning_rate", type=float, default=2e-2, help="Learning rate to use for the optimization")
    parser.add_argument('-m', "--momentum", type=float, default=0.9, help='Momentum to use for the optimization')
    parser.add_argument("-op",  "--optimizer", type=str, default="adam", help="The optimizer to use")
    
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="Verbose output")
    parser.add_argument("-vdl", "--vis_step", type=int, default=1000, help="Visualize the result every k epochs")
    parser.add_argument("-vis", "--visualize", action="store_true", help="Visualize the results")
    parser.add_argument("--vis_dir", type=str, default='out_img', required=False, help="Image directory")
    
    # Shared Arguments
    parser.add_argument("--joint_rec_epochs", type=int, default=1, required=False, help="How many epochs to reconstruct together")
    parser.add_argument("--k_batches", type=int, default=1, required=False, help="Number of batches to attack simultaneously")
    parser.add_argument("-b", "--batch_size", type=int, default=32, required=False, help="Batch size to use")
    parser.add_argument("-l", "--loss", type=str, default="CE", help="Loss function to use")
   
    # Label restore
    parser.add_argument("-res_label", "--restore_label", type=str, choices=['aaai_st_full', 'aaai_en_full','aaai_avg_full', 'aaai_st_only', 'aaai_en_only','aaai_avg_only'], default=None, help="Which type of label attack to use")
    parser.add_argument('-label_samp', '--restore_label_samp', type=int, default=0, required=False, help='If specified label restoration with this many dummy inputs is initiated')


    # Attack arguments
    parser.add_argument("-reorder_prior", "--reorder_prior", type=str, choices=['l2_mean_conv', 'l2_max_conv', 'l1_mean_conv', 'l1_max_conv', 'l2_mean', 'l2_max', 'l1_mean', 'l1_max'], default='none', help="Loss to be used for the attack")
    parser.add_argument("-at",  "--attack_type", type=str, required=False, help="Unused atm.")
    parser.add_argument("-ar",  "--attack_repeated_init", type=int, default=1,
                        help="The amount of times the random attack init pattern should be repeated. Should divide the image size.")
    parser.add_argument("-atv", "--attack_total_variation", type=float, default=5e-4, help="Adds a total variation regulation term at the end")
    parser.add_argument("-al2", "--attack_l2_reg", type=float, default=0, help="Adds a network accuracy regulation term at the end")
    parser.add_argument("-aac", "--attack_accuracy_reg", type=float, default=0,
                        help="Adds a network accurarcy regulation term at the end (https://arxiv.org/pdf/2004.10397.pdf)")
    # Defense arguments
    parser.add_argument("-dai", "--defense_attack_iterations", type=int, default=2000, help="The amount of attack opt. steps to take during the defense.")
    parser.add_argument("-dii", "--defense_inner_iterations", type=int, default=1, help="The amount of inner attack iterations at every step of the defense")
    parser.add_argument("-dd", "--defense_dilation", type=int, default=500, help="The amount of outer training iterations before we have an inner defense run.")
    parser.add_argument('--flax', action='store_true', help='Whether to run flax')
    parser.add_argument('--debug', action='store_true', help='Disable jit')
    
    #Neptune args
    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', required=False, default=None)
    parser.add_argument('--neptune-token', type=str, help='neptune token', required=False, default=None)
    parser.add_argument('--neptune-label', type=str, help='name of the run', required=False,  default=None)

    
    if cmd is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd)
    
    if args.same_cls is None:
        args.same_cls = False
    if args.rand_batch is None:
        args.rand_batch = False
    assert not (args.same_cls) or args.dataset == 'FEMNIST'
    assert (args.max_batch is None) or args.dataset == 'FEMNIST'
    assert 'many' in args.fedavg or args.reorder_prior == 'none'
    assert not args.fedavg.endswith('known_labels') or args.restore_label is None
    
    assert (args.neptune is None or not(args.neptune_token is None or args.neptune_label is None) )

    return args
