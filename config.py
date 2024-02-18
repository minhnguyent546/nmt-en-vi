
def get_config():
    config = {
        'dataset_path': 'mt_eng_vietnamese',
        'dataset_name': 'iwslt2015-en-vi',
        'checkpoints_dir': 'checkpoints',
        'tokenizer_basename': 'tokenizer_{}.json',
        'data_loaders_basename': 'data_loaders.pt',
        'model_dir': 'weights',
        'model_basename': 'transformer',
        'preload': None,
        'experiment_name': 'runs/model',
        'src_lang': 'en',
        'target_lang': 'vi',
        'batch_size': 15,
        'beam_size': 5,
        'num_epochs': 20,
        'num_eval_samples': 5,
        'learning_rate': 1e-4,
        'seq_length': 80,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ffn': 2048,
        'dropout_rate': 0.1,
        'grad_clipping': 1.0,
    }
    return config
