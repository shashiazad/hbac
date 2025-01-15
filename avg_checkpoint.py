import torch
import argparse
import collections
import os
import re

import torch

import logging
import os
import shutil
from typing import List, Optional



def average_checkpoints(inputs,fold):
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: torch.serialization.default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params


    torch.save(averaged_params, './avg_fold%d.pth'%(fold))
    return new_state




data_dir='./models'
weights=os.listdir(data_dir)

weights=[os.path.join(data_dir,x) for x in weights]

grouped_files = [[] for _ in range(10)]


for file in weights:
    fold_num = int(file.split('_')[0][-1])  # 获取文件名中的fold编号

    epoch_num=int(file.split('_')[2])  # 获取文件名中的fold编号



    grouped_files[fold_num].append(file)  # 将文件名添加到对应的组中


for i in range(len(grouped_files)):


    grouped_files[i].sort(key=lambda x:x.split('_')[-1].rsplit('.',1)[0])

    print(grouped_files[i])
    average_checkpoints(grouped_files[i][:3],i)