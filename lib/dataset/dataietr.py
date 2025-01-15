import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4


import ast
import pickle
import random


import mne
import numpy as np
import copy
import pandas as pd


from lib.utils.logger import logger

import albumentations as A


from scipy.signal import butter, lfilter

from train_config import config as cfg
class AlaskaDataIter():
    def __init__(self, df,
                 training_flag=True, shuffle=True, use_eeg=True, use_spec=False, use_mix=False):

        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None  ##decided by self.parse_file

        self.df = df

        #
        logger.info(' contains%d samples' % len(self.df))
        self.train_trans = A.Compose([

            A.HorizontalFlip(p=0.5)

        ])

        TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
        self.TARS2 = {x: y for y, x in TARS.items()}

        self.eeg_nms = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                        'Fz', 'Cz', 'Pz',
                        'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
                        'EKG']

        # with open('specs_dict.pkl', mode='rb') as f:
        #     self.all_specs = pickle.load(f)

        self.LL = ['Fp1', 'F7', 'T3', 'T5', 'O1']

        self.RL = ['Fp2', 'F8', 'T4', 'T6', 'O2']

        self.LP = ['Fp1', 'F3', 'C3', 'P3', 'O1']

        self.RP = ['Fp2', 'F4', 'C4', 'P4', 'O2']

        self.mid = ['Fz', 'Cz', 'Pz']

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg
        self.use_spec = use_spec
        self.use_mix = use_mix
        if self.use_spec or self.use_mix:
            with open('specs_dict.pkl', mode='rb') as f:
                self.all_specs = pickle.load(f)

    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):

        return len(self.df)

    def avg_lead(self, waves):

        # copy一份，防止原地修改
        waves = copy.deepcopy(waves)

        meadn = np.mean(waves[:19, :], axis=0)
        data = waves[:19, :] - meadn

        return data

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)

        brain_leads = [self.LL, self.RL, self.LP, self.RP]

        leads = []

        for chain in brain_leads:
            for i in range(len(chain) - 1):
                tmp_lead = waves[self.leads_dict[chain[i]]] - waves[self.leads_dict[chain[i + 1]]]
                leads.append(tmp_lead)

        data = np.concatenate([leads], axis=0)

        return data
    def brain_lead_reverse(self, waves):
        waves = copy.deepcopy(waves)

        LL_r=self.LL[::-1]
        RL_r = self.RL[::-1]
        LP_r = self.LP[::-1]
        RP_r = self.RP[::-1]

        brain_leads = [LL_r, RL_r, LP_r, RP_r]

        leads = []

        for chain in brain_leads:
            for i in range(len(chain) - 1):
                tmp_lead = waves[self.leads_dict[chain[i]]] - waves[self.leads_dict[chain[i + 1]]]
                leads.append(tmp_lead)

        data = np.concatenate([leads], axis=0)

        return data
    def mirror_spec(self, data):

        # index_choice = [[0, 1, 3, 2], [0, 1, 2, 3], [1, 0, 2, 3], [1, 0, 3, 2]]
        indx = [1, 0, 3, 2]
        return data[..., indx]

    def mirror_eeg(self, data):

        indx1 = [0, 1, 2, 3, 4, 5, 6, 7]
        indx2 = [11, 12, 13, 14, 15, 16, 17, 18]
        data[indx1, ...], data[indx2, ...] = data[indx2, ...], data[indx1, ...]

        return data

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype="band")

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def get_eeg(self, dp, is_training):
        eeg_path = '../hms-harmful-brain-activity-classification/train_eegs/%s.parquet' % (dp['eeg_id'])
        eeg = pd.read_parquet(eeg_path)

        eeg_label_offset_list = dp['eeg_offset_list']

        eeg_label_offset_list = ast.literal_eval(eeg_label_offset_list)
        eeg_label_offset_list = [int(x) for x in eeg_label_offset_list]

        offset_min = np.min(eeg_label_offset_list)
        offset_max = np.max(eeg_label_offset_list)

        vote_count=dp['vote_count']
        vote_count = ast.literal_eval(vote_count)
        vote_count = [int(x) for x in vote_count]


        if is_training:
            n=len(eeg_label_offset_list)
            random_index = random.randint(0,n-1)
            offset=eeg_label_offset_list[random_index]

            weights = vote_count[random_index] / 20.
            if cfg.TRAIN.stage ==2:
                if vote_count[random_index]>6:
                    weights=1
                else:
                    weights=0

        else:
            offset = int((offset_min + offset_max) // 2)

            vote_count=np.mean(vote_count)

            if vote_count<6:
                weights=0
            else:
                weights=1
        # print(offset)
        # if random.uniform(0, 1) < 1. and is_training:
        #     offset += random.uniform(-1, 1)
        #     offset = np.clip(offset, a_min=offset_min, a_max=offset_max)

        eeg = eeg.iloc[int(offset * 200):int(offset * 200) + 10000]

        waves = eeg.values

        waves = np.transpose(waves, axes=[1, 0])

        ## 0.5 p mirror
        if random.uniform(0, 1) < 0.5 and is_training:
            waves = self.mirror_eeg(waves)

        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0

        waves = self.brain_lead(waves)
        waves = np.array(waves, dtype=np.float64)

        waves = np.clip(waves, -1024, 1024)

        waves = mne.filter.filter_data(waves, 200, 0.5, 20, verbose=False)

        # waves = self.butter_bandpass_filter(waves, 0.5, 20, 200, 2)

        return waves,weights

    def get_spec(self, dp, is_training):
        # eeg_path = '../hms-harmful-brain-activity-classification/train_spectrograms/%s.parquet' % (dp['spec_id'])
        # spec = pd.read_parquet(eeg_path)
        spec = self.all_specs[str(dp['spec_id'])]
        # spec=spec.values[:,1:]



        spec_offset_list = dp['spec_offset_list']

        spec_offset_list = ast.literal_eval(spec_offset_list)
        spec_offset_list = [int(x) for x in spec_offset_list]

        offset_min = np.min(spec_offset_list)
        offset_max = np.max(spec_offset_list)

        vote_count = dp['vote_count']
        vote_count = ast.literal_eval(vote_count)
        vote_count = [int(x) for x in vote_count]

        images = []

        if is_training:
            n=len(spec_offset_list)
            random_index = random.randint(0,n)
            r=spec_offset_list[random_index]//2

            weights=vote_count[random_index]/20.
            # offset = int((offset_min + offset_max) // 2)
        else:
            r = int((offset_min + offset_max) // 4)
            weights=1

        # random shift in time axis
        # if random.uniform(0, 1) < 1. and is_training:
        #     r += random.uniform(-10, 10)
        #     r = np.clip(r, a_min=0, a_max=max_time - 300)
        #     r = int(r)

        for region in range(4):
            img = spec[r:r + 300, region * 100:(region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # STANDARDIZE PER IMAGE
            # ep = 1e-6
            # m = np.nanmean(img.flatten())
            # s = np.nanstd(img.flatten())
            # img = (img - m) / (s + ep)
            img = np.nan_to_num(img, nan=0.0)

            images.append(img)

        images = np.stack(images, -1)

        if is_training:
            data = self.train_trans(image=images)
            images = data['image']
            ## 0.5 p mirror
            if random.uniform(0, 1) < 0.5:
                images = self.mirror_spec(images)

        data = np.transpose(images, [2, 0, 1])

        return data,weights

    def get_mix(self, dp, is_training):
        flip = False
        if is_training and random.uniform(0, 1) < 0.5:
            flip = True

        eeg_path = '../hms-harmful-brain-activity-classification/train_eegs/%s.parquet' % (dp['eeg_id'])
        eeg = pd.read_parquet(eeg_path)

        eeg_label_offset_list = dp['eeg_offset_list']

        eeg_label_offset_list = ast.literal_eval(eeg_label_offset_list)
        eeg_label_offset_list = [int(x) for x in eeg_label_offset_list]
        offset_min = np.min(eeg_label_offset_list)
        offset_max = np.max(eeg_label_offset_list)

        vote_count = dp['vote_count']
        vote_count = ast.literal_eval(vote_count)
        vote_count = [int(x) for x in vote_count]

        n = len(eeg_label_offset_list)
        random_index = random.randint(0, n)
        if is_training:
            offset = eeg_label_offset_list[random_index]
            weights = vote_count[random_index] / 20.
        else:
            offset = int((offset_min + offset_max) // 2)
            weights = 1

        eeg = eeg.iloc[int(offset * 200):int(offset * 200) + 10000]

        waves = eeg.values

        waves = np.transpose(waves, axes=[1, 0])
        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0

        if flip:
            waves = self.mirror_eeg(waves)

        waves = self.brain_lead(waves)
        waves = np.array(waves, dtype=np.float64)

        waves = np.clip(waves, -1024, 1024)

        waves = mne.filter.filter_data(waves, 200, 0, 40, verbose=False)
        #
        # get spec
        spec_offset_list = dp['spec_offset_list']
        spec_offset_list = ast.literal_eval(spec_offset_list)
        spec_offset_list = [int(x) for x in spec_offset_list]

        spec_offset_min = np.min(spec_offset_list)
        spec_offset_max = np.max(spec_offset_list)

        if is_training:
            r = spec_offset_list[random_index] // 2
        else:
            r = int((spec_offset_min + spec_offset_max) // 4)

        # eeg_path = '../hms-harmful-brain-activity-classification/train_spectrograms/%s.parquet' % (dp['spec_id'])
        # spec = pd.read_parquet(eeg_path)
        # spec=spec.values[:,1:]
        spec = self.all_specs[str(dp['spec_id'])]

        images = []
        for region in range(4):
            img = spec[r:r + 300, region * 100:(region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # STANDARDIZE PER IMAGE
            # ep = 1e-6
            # m = np.nanmean(img.flatten())
            # s = np.nanstd(img.flatten())
            # img = (img - m) / (s + ep)
            img = np.nan_to_num(img, nan=0.0)

            images.append(img)

        images = np.stack(images, -1)

        if flip:
            images = self.mirror_spec(images)

        images = np.transpose(images, [2, 0, 1])

        return waves, images, weights

    def single_map_func(self, dp, is_training):
        """Data augmentation function."""
        ####customed here

        if self.use_eeg:
            eeg_data,weights = self.get_eeg(dp, is_training)
        if self.use_spec:
            spec_data,weights = self.get_spec(dp, is_training)
        if self.use_mix:
            spec_data, eeg_data = self.get_mix(dp, is_training)
        label = dp[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].values

        label = np.array(label, dtype=np.float32)

        return eeg_data, weights, label
