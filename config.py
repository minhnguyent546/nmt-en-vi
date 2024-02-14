from pathlib import Path

def get_config():
    config = {
        'dataset_path': 'mt_eng_vietnamese',
        'dataset_name': 'iwslt2015-en-vi',
        'tokenizer_dir': 'tokenizers',
        'tokenizer_basename': 'tokenizer_{}.json',
        'model_dir': 'weights',
        'model_basename': 'transformer',
        'preload': None,
        'experiment_name': 'runs/model',
        'src_lang': 'en',
        'target_lang': 'vi',
        'batch_size': 15,
        'num_epochs': 20,
        'log_step': 5,
        'num_eval_samples': 5,
        'learning_rate': 1e-4,
        'seq_length': 80,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ffn': 2048,
        'dropout_rate': 0.1,
    }
    return config

def get_weights_file_path(epoch: str, config: dict) -> str:
    model_dir = config['model_dir']
    model_basename = config['model_basename']
    model_file = f"{model_basename}_{epoch}.pt"
    return str(Path(model_dir) / model_file)
