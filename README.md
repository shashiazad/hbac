# HMS-eeg-comp
2nd solution to HMS - Harmful Brain Activity Classification

## 1. data

Unzip the hms data to ../hms-harmful-brain-activity-classification

Produce new train.csv by
```
python reconstructed_target.py
```

## 2. stage1 train

#### 2.1  define model 
The code import pytorch model define lib from lib.core.base_trainer.model, 
please modify the model in lib.core.base_trainer.model.py

#### 2.2 train
```python train.py```


## 3. stage2 train
#### 3.1 modify config
```
config.TRAIN.stage=2
config.TRAIN.epoch = 5
config.TRAIN.init_lr=0.0001
```

#### 3.2 average the best 3 weights per fold by run

``` python avg_checkpoint.py ```
It will produce avg_fold0.pth .....avg_fold9.pth, idon't know how much it help to use a checkpoints average here.

#### 
```python train.py```

#### choose the best metric weights