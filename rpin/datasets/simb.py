import cv2
import torch
import hickle
import pickle
import numpy as np
from glob import glob

from rpin.datasets.phys import Phys
from rpin.utils.misc import tprint
from rpin.utils.config import _C as C
from PIL import Image
import os


class SimB(Phys):
    def __init__(self, data_root, split, image_ext='.jpg'):
        super().__init__(data_root, split, image_ext)

        self.video_list = sorted(glob(f'{self.data_root}/{self.split}/*/'))
        self.anno_list = [v[:-1] + '.pkl' for v in self.video_list]

        env_meta_file_path = os.path.join(self.data_root, '{}_env_meta.pkl'.format(self.split))
        self.env_meta = None

        if C.INPUT.PRELOAD_TO_MEMORY:
            print('loading data from hickle file...')
            data = hickle.load(f'{self.data_root}/{self.split}.hkl')
            self.total_img = np.transpose(data['X'], (0, 1, 4, 2, 3))
            self.total_box = np.zeros((data['y'].shape[:3] + (5,)))
            for anno_idx, anno_name in enumerate(self.anno_list):
                tprint(f'loading progress: {anno_idx}/{len(self.anno_list)}')
                with open(anno_name, 'rb') as f:
                    boxes = pickle.load(f)
                self.total_box[anno_idx] = boxes

        self.video_info = np.zeros((0, 2), dtype=np.int32)
        for idx, video_name in enumerate(self.video_list if not C.INPUT.PRELOAD_TO_MEMORY else self.total_box):
            tprint(f'loading progress: {idx}/{len(self.video_list)}')
            if C.INPUT.PRELOAD_TO_MEMORY:
                num_sw = self.total_box[idx].shape[0] - self.seq_size + 1
            else:
                if C.RPIN.TRAIN_MODE == 'env_mask':
                    num_im = len(glob(f'{video_name}/*_bmask{image_ext}'))
                elif image_ext == '.pkl':
                    num_im = len(glob(f'{video_name}/*_data{image_ext}'))
                else:
                    num_im = len(glob(f'{video_name}/*{image_ext}'))
                num_sw = num_im - self.seq_size + 1  # number of sliding windows

            if num_sw <= 0:
                continue
            video_info_t = np.zeros((num_sw, 2), dtype=np.int32)
            video_info_t[:, 0] = idx  # video index
            video_info_t[:, 1] = np.arange(num_sw)  # sliding window index
            self.video_info = np.vstack((self.video_info, video_info_t))

        if os.path.exists(env_meta_file_path):
            self.env_meta = self.read_pickle_data(env_meta_file_path)

    def _parse_image(self, video_name, vid_idx, img_idx):
        if C.INPUT.PRELOAD_TO_MEMORY:
            data = self.total_img[vid_idx, img_idx:img_idx + self.input_size].copy()
        else:
            if C.RPIN.TRAIN_MODE == 'env_mask':
                image_list = sorted(glob(f'{video_name}/*_bmask{self.image_ext}'))
            elif self.image_ext == '.pkl':
                image_list = sorted(glob(f'{video_name}/*_data{self.image_ext}'))
            else:
                image_list = sorted(glob(f'{video_name}/*{self.image_ext}'))
            image_list = image_list[img_idx:img_idx + self.input_size]
            data = np.zeros((self.input_size, 3, self.input_height, self.input_width))
            if self.image_ext == '.jpg' or self.image_ext == '.png':  # RealB Case
                data = np.array([
                    np.asarray(Image.open(image_name)) for image_name in image_list
                ], dtype=np.float).transpose((0, 3, 1, 2))
            elif self.image_ext == '.pkl':
                data = np.array([self.read_pickle_data(image_name) for image_name in image_list],
                                dtype=np.float).transpose(0, 3, 1, 2)
            else:
                raise RuntimeError('Not Supporting Image Ext Name')
            for c in range(C.INPUT.IMAGE_CHANNEL):
                data[:, c] -= C.INPUT.IMAGE_MEAN[c]
                data[:, c] /= C.INPUT.IMAGE_STD[c]

        return data

    def _parse_label(self, anno_name, vid_idx, img_idx):
        if C.INPUT.PRELOAD_TO_MEMORY:
            boxes = self.total_box[vid_idx, img_idx:img_idx + self.seq_size, :, 1:].copy()
        else:
            with open(anno_name, 'rb') as f:
                boxes = pickle.load(f)[img_idx:img_idx + self.seq_size, :, 1:]
        gt_masks = np.zeros((self.pred_size, boxes.shape[1], C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE))
        return boxes, gt_masks

    def _parse_mask(self, video_name, vid_idx, img_idx, boxes, data):
        if C.RPIN.MASK_MODE == 'gt_mask' or C.RPIN.MASK_MODE == 'kmean_mask':
            if C.RPIN.MASK_MODE == 'gt_mask':
                mask_list = sorted(glob(f'{video_name}/*_bmask.pkl'))
            elif C.RPIN.MASK_MODE == 'kmean_mask':
                mask_list = sorted(glob(f'{video_name}/*_kmean_mask.pkl'))
            mask_list = mask_list[img_idx:img_idx + self.input_size]
            data = np.zeros((self.input_size, 2, self.input_height, self.input_width))
            if len(mask_list) > 0:
                data = np.array([self.read_pickle_data(mask_name) for mask_name in mask_list],
                                        dtype=np.float).transpose(0, 3, 1, 2)
        else:
            NotImplemented
        return data


    def read_pickle_data(self, pickle_data_path):
        with open(pickle_data_path, 'rb') as f:
            data = pickle.load(f)
        return data
