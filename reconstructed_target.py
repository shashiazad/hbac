import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('../hms-harmful-brain-activity-classification/train.csv')
TARGETS = df.columns[-6:]
new_target = []

print(len(np.unique(df['label_id'])),'labelid')
print(len(np.unique(df['label_id'])),'labelid')

train = copy.deepcopy(df)

train['gg'] = -1
train.loc[0, 'gg'] = 0

eeg_ids=np.unique(train['eeg_id'])

eeg_ids.sort()
cols=['eeg_id','spec_id', "patient_id",'eeg_offset_list','spec_offset_list','target']+TARGETS.to_list()

dst = pd.DataFrame(columns=cols)

agree=0
for eegid in tqdm(eeg_ids):
    same_eeg_id_indx= train['eeg_id']==eegid

    y_data = train.loc[same_eeg_id_indx][TARGETS].values


    total_evaluators = train.loc[same_eeg_id_indx][
        ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1).to_list()


    vote= y_data.sum(axis=0, keepdims=True)

    vote= vote/np.sum(vote)

    vote=vote[0]

    eeg_offset_list = train.loc[same_eeg_id_indx]['eeg_label_offset_seconds'].to_list()
    spec_offset_list = train.loc[same_eeg_id_indx]['spectrogram_label_offset_seconds'].to_list()


    eeg_offset_list=train.loc[same_eeg_id_indx]['eeg_label_offset_seconds'].to_list()
    spec_offset_list = train.loc[same_eeg_id_indx]['spectrogram_label_offset_seconds'].to_list()

    spectrogram_id= train.loc[same_eeg_id_indx]['spectrogram_id'].value_counts().idxmax()

    target = train.loc[same_eeg_id_indx]['expert_consensus'].value_counts().idxmax()

    patient_id = train.loc[same_eeg_id_indx]['patient_id'].value_counts().idxmax()

    data = {'eeg_id': eegid,
            'spec_id':spectrogram_id,
            'patient_id':patient_id,
            'eeg_offset_list': eeg_offset_list,
            'spec_offset_list':spec_offset_list,
            'target':target,
            'seizure_vote':vote[0],
            'lpd_vote':vote[1],
            'gpd_vote':vote[2],
            'lrda_vote':vote[3],
            'grda_vote':vote[4],
            'other_vote':vote[5],
            'vote_count':total_evaluators
    }

    dst = dst._append(data, ignore_index=True)

dst.to_csv('train.csv',index=False)
