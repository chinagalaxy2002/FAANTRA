import argparse

def get_args():
    # Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('--seed', type=int, default=42)
    # From FUTR
    parser.add_argument("--cpu", action='store_true', help='run in cpu')
    parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoint to load. Only use if planning to load a checkpoint.")
    parser.add_argument("--wandb-new-id", action='store_true', help="Force wandb to create a new run in case of loading from checkpoint.")
    return parser.parse_args()

# Update arguments with config file
# Check README in config/ folder for more details
def update_args(args, config):
    # Directories
    args.frame_dir = config['frame_dir']                    # Directory where frames are stored
    args.save_dir = config['save_dir'] + '/' + args.model   # Directory to save model checkpoints, predictions, etc
    args.store_dir = config['store_dir']                    # Directory to save dataset information, such as clip splits

    args.store_mode = config['store_mode']  # Store splits or load dataset
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']      # Full length of the clip. Includes observation and anticipation frames
    args.dataset = config['dataset']        # Dataset name
    args.radi_smoothing = config['radi_smoothing']  # Radius of the label smoothing used in the segmentation task
    args.epoch_num_frames = config['epoch_num_frames']  # Number of frames to train on each epoch
    args.feature_arch = config['feature_arch']  # Name of the backbone feature extractor. Supports rny002, rny004, rny006 and rny008
    args.warm_up_epochs = config['warm_up_epochs']  # Number of epochs to warm up the learning rate
    # Temporal architecture arguments
    args.temporal_arch = config['temporal_arch']    # Which temport architecture to use. Supports "ed_sgp_mixer" and "none"
    args.n_layers = config['n_layers']  # Number of layers to use in temporal architecture. Does not apply to the transformer
    args.sgp_ks = config['sgp_ks']  # Kernel size of the SGP and SGP-Mixer layers
    args.sgp_r = config['sgp_r']    # r factor in SGP and SGP-Mixer layers
    # --- 新增：读取对比学习权重 ---
    # 如果配置文件里没有写，默认给 0.1
    args.contrastive_weight = config.get("contrastive_weight", 0.1)
    args.num_workers = config['num_workers']    # Number of workers to use for data loading
    # Load the dual training arguments
    if 'jointtrain' in config and config['use_jointtrain'] :
        args.jointtrain = config['jointtrain']
    else:
        args.jointtrain = None
    args.wandb_mode = config['wandb_mode']  # Wandb logging mode. Can be "online", "offline" or "disabled"

    # Arguments for FUTR compatibility
    args.epochs = config["num_epochs"]
    args.lr = config["learning_rate"]                   # FUTR's default is 0.001, while T-Deed's is 0.0008
    # T-Deed just used default weight decay
    args.weight_decay = config["weight_decay"]
    args.obs_perc = config["obs_perc"]      # Percentage of clip_len to use for observation
    args.pred_perc = config["pred_perc"]    # Percentage of clip_len to anticipate. Anticipates frames directly after observed frames
    args.n_query = config["n_query"]        # Number of queries to use in the transformer. This determines the max number of actions the model can predict
    args.seg = config["seg"]                # Whether to perform the segmentation auxilary task or not
    args.anticipate = config["anticipate"]  # Whether to perform anticipation or not
    # Hard coded
    args.pos_emb = True # config["pos_emb"]
    args.max_pos_len = 2000 # config["max_pos_len"]

    # FUTR transformer model params
    args.n_head = config["n_head"]
    args.hidden_dim = config["hidden_dim"]
    args.n_encoder_layer = config["n_encoder_layer"]
    args.n_decoder_layer = config["n_decoder_layer"]

    # Extra arguments

    # So far only supports excluding all classes after a certain number
    # Stuff like [5, 7, 9, 12] will not work, as the rest of the code is not prepared for it and loss calcualtions will break
    # If you want to exclude specific classes, then you need to change the classes order in data/{dataset_name}/classes.txt
    # and put the classes you want to exclude at the end of the list
    # Keep in mind that changing the classes order will make your output incompatible with the BAA challenge
    args.excluded_classes = config['excluded_classes']
    args.class_weights = config['class_weights'] # A list with length of n_classes (Includes background).
    args.eos_weight = config['eos_weight']  # Weight for the end of sequence token if used
    args.loss_func = config.get("loss_func", "CE") # Loss function to use. Default is CrossEntropy
    args.offset_loss_weight = config.get("offset_loss_weight", 1.0) # Weight for the offset loss
    args.test_obs_perc = config["test_obs_perc"]    # Observation percentage to use when evaluating. This way tests with different observations can be carried. Only one percentage at a time

    # Arguments to toggle a cheating dataset. A cheating dataset is a dataset that provides some of the anticipation frames to the model.
    # Essentially creating a cheating model. This was used for tests that did not make it into the paper. Feel free to ignore
    args.cheating_dataset = config.get("cheating_dataset", False)
    # What percentage of the video to provide when cheating. Can technically only provide anticipation frames with no observation frames.
    args.cheating_range = config.get("cheating_range", [0.0, 1.0])
    if args.cheating_dataset:
        assert args.cheating_range is not None, "Cheating range not provided"
        assert isinstance(args.cheating_range, list), "Cheating range should be a list"
        assert args.cheating_range[0] < args.cheating_range[1], "Smallest number in cheating range should be first"
        assert args.cheating_range[1] <= 1.0, "Cheating range end should not be greater than 1.0"
        assert isinstance(args.cheating_range[0], float) and isinstance(args.cheating_range[1], float), "Cheating range should be floats"
    
    args.CALF_matching = config.get("CALF_matching", False) # Use CALF matching for loss calculation. Can also be called offset matching
    args.CALF_probability_matching = config.get("CALF_probability_matching", False) # Does CALF matching with probability instead of offset
    if args.CALF_probability_matching:
        assert args.CALF_matching, "CALF probability matching requires CALF matching to be enabled"
    args.anticipate_background = config.get("anticipate_background", False) # Anticipate background class instead of EOS. Still uses EOS weight for loss weighting
    args.actionness = config.get("actionness", False) # Use actionness (objectness equivialent) instead of EOS. EOS weight is ignored
    args.use_anchors = config.get("use_anchors", False) # Use anchors for anticipation instead of predicting the frame
    assert not (args.anticipate_background and args.actionness), "Can't anticipate background and use actionness at the same time"
    if args.use_anchors: assert (args.use_anchors and (args.anticipate_background or args.actionness)), "Cannot use anchors with EOS, needs to use background or actionness"
    args.start_map_epoch = config.get("start_map_epoch", 20) # Which epoch to start saving checkpoints and running mAP calculations on validation set
    args.mask_attn = config.get("mask_attn", False) # Mask attention for the transformer
    args.mask_attn_window_src = config.get("mask_attn_window_src", 0) # Window size for encoder attention masking
    args.mask_attn_window_tgt = config.get("mask_attn_window_tgt", 0) # Window size for decoder attention masking
    
    return args