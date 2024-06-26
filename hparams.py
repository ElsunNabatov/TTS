
def create_hparams():
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "training_model": "vits",
        "epochs": 100,
        "seed":1234,
        "dynamic_loss_scaling":True,
        "fp16_run": False,
        #"device" : "cuda",
        "csv_path" : "datasets/voice_book/csvs/",
        "device" : "cuda",
        ################################
        # Data Parameters             #
        ################################
        "text_cleaners" : ['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value": 32768.0,
        "sampling_rate": 22050,
        "filter_length":1024,
        "hop_length":256,
        "win_length":1024,
        "n_mel_channels":80,
        "mel_fmin":0.0,
        "mel_fmax":8000.0,
        "segment_length": 16000,

        ################################
        # Model Parameters             #
        ################################
        "n_symbols":128,
        "symbols_embedding_dim":512,

        # Encoder parameters
        "encoder_kernel_size":5,
        "encoder_n_convolutions":3,
        "encoder_embedding_dim":512,

        # Decoder parameters
        "n_frames_per_step":1,  # currently only 1 is supported
        "decoder_rnn_dim":1024,
        "prenet_dim":256,
        "max_decoder_steps":1000,
        "gate_threshold":0.5,
        "p_attention_dropout":0.1,
        "p_decoder_dropout":0.1,

        # Attention parameters
        "attention_rnn_dim":1024,
        "attention_dim":128,

        # Location Layer parameters
        "attention_location_n_filters":32,
        "attention_location_kernel_size":31,

        # Mel-post processing network parameters
        "postnet_embedding_dim":512,
        "postnet_kernel_size":5,
        "postnet_n_convolutions":5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate":False,
        "learning_rate":1e-3,
        "weight_decay":1e-6,
        "grad_clip_thresh":1.0,
        "batch_size":1,
        "mask_padding":True,  # set model's padded outputs to padded values
        "learning_rate_wv": 1e-4,
        "sigma": 1.0,
        "iters_per_checkpoint": 2000,
        "n_flows": 12,
        "n_group": 8,
        "n_early_every": 4,
        "n_early_size": 2,
        "n_layers": 8,
        "n_channels": 256,
        "kernel_size": 3
        
    }

    return hparams