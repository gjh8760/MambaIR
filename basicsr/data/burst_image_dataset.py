from torch.utils import data as data

from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import os
import pickle as pkl
import torch
import cv2
import numpy as np


@DATASET_REGISTRY.register()
class ZurichRawBurstDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_folder = opt['dataroot']
        self.io_backend_opt = opt['io_backend']
        self.processing = opt['processing']  # processing 객체가 외부에서 주입된다고 가정
        self.file_client = None

        self.paths = sorted([
            os.path.join(self.gt_folder, f)
            for f in os.listdir(self.gt_folder)
            if f.lower().endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        gt_path = self.paths[idx]
        frame = cv2.cvtColor(cv2.imread(gt_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        data = {'frame': frame}
        data = self.processing(data)

        return {
            'lq': data['burst'],      # synthetic RAW burst
            'gt': data['frame_gt'],   # ground truth
            'meta_info': data['meta_info'],
            'gt_path': gt_path
        }


@DATASET_REGISTRY.register()
class ZurichRawBurstValDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        self.gt_paths = sorted([
            os.path.join(self.gt_folder, f)
            for f in os.listdir(self.gt_folder)
        ])
        self.lq_paths = sorted([
            os.path.join(self.lq_folder, f)
            for f in os.listdir(self.lq_folder)
        ])

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.gt_paths[idx], 'im_rgb.png')
        gt_frame = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        gt_frame = torch.from_numpy(gt_frame.astype(np.float32)).permute(2, 0, 1).float() / (2**14)
        
        lq_paths = sorted([
            os.path.join(self.lq_paths[idx], f)
            for f in os.listdir(self.lq_paths[idx])
            if f.lower().endswith(('.jpg', '.png'))
        ])
        lq_frames = torch.stack([
            torch.from_numpy(cv2.imread(lq_path, cv2.IMREAD_UNCHANGED).astype(np.float32)).permute(2, 0, 1).float() / (2**14)
            for lq_path in lq_paths
        ], dim=0)

        meta_info_path = os.path.join(self.gt_paths[idx], 'meta_info.pkl')
        meta_info = pkl.load(open(meta_info_path, 'rb', -1))

        return {'lq': lq_frames, 'gt': gt_frame, 'meta_info': meta_info, 'gt_path': gt_path, 'lq_path': lq_paths}
        