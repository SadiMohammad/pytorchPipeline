# pytorchPipeline
pytorch pipeline for segmentation model

## Project Structure

```
- ckpts
- configs
- dataloaders
    -- datasplit
    -- dataloader.py
- logs
- models
    -- layers
    -- model.py
- utils
    -- logger.py
    -- losses.py
    -- metric.py
    -- save_config.py
    -- transforms.py
- train.py
- train.sh
- trainer.py
```
## Training

Current this code base works for Python version >= 3.8

First, you have to prepare a dataset where dataset is splited into train, validation and test set. Keep the ids of this files in the respective txt files in `./dataloaders/datasplit`. </p>
Train model by running the following commands.

```
cd pytorchPipeline
./train.sh
```

### Current Version
- [x] Binary Segmentation
- [x] Dataloader for custom dataset
- [x] Dataloader through RAM added
- [ ] Multiclass Segmentation
- [x] Local logging
- [x] Wandb added for remote logging and visualization
- [ ] Tensorboard