from matplotlib import pyplot as plt

from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import AlaskaDataIter
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,GroupKFold

from train_config import config as cfg
from lib.core.kaggle.kl import score
import setproctitle



setproctitle.setproctitle("comp")



def main():
    n_fold=10
    def get_fold(n_fold=n_fold):


        data=pd.read_csv(cfg.DATA.data_file)


        folds = data.copy()
        gkf = GroupKFold(n_splits=n_fold)

        for n, (train_index, val_index) in enumerate(gkf.split(folds, folds.target,folds.patient_id)):
            folds.loc[val_index, 'fold'] = int(n)
        return folds


    data=get_fold(n_fold)

    all_oof = []
    all_true = []
    all_weight = []
    for fold in range(n_fold):
        ###build dataset

        train_ind = data[data['fold'] != fold].index.values
        train_data = data.iloc[train_ind].copy()
        val_ind = data[data['fold'] == fold].index.values
        val_data = data.iloc[val_ind].copy()



        # add 1000data not that important
        train_data=train_data._append(val_data.iloc[:1000])

        val_data=val_data[1000:]

        ###build trainer
        trainer = Train(train_df=train_data,
                        val_df=val_data,
                        fold=fold)

        ### train
        trainer.custom_loop()

        all_oof.append(trainer.oof_pre)
        all_true.append(trainer.oof_gt)
        all_weight.append(trainer.oof_weight)

    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)
    all_weight = np.concatenate(all_weight)

    bool_indx=all_weight==1

    all_oof=all_oof[bool_indx]
    all_true = all_true[bool_indx]

    oof = pd.DataFrame(all_oof.copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true['id'] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name='id')
    print('CV Score KL-Div =',cv)

    




if __name__=='__main__':
    main()