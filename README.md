# Machine translation (en-vi) using Transformer model

## Setup for NMT from english to vietnamse task

- Preprocessing data:
```bash
python preprocess_nmt.py --config 'config/config.yaml'
```

- To train the model:
```bash
python train_nmt.py --config 'config/config.yaml'
```

- To test the model:
```bash
python test_nmt.py --config 'config/config.yaml'
```

## TODO
- [x] Expand English contractions
- [ ] Process result sentences (e.g. capitalization, punctuation)
