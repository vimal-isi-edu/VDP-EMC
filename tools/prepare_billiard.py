import os
import cv2
import pickle
import hickle
import argparse
import numpy as np
from PIL import Image


def arg_parse():
    parser = argparse.ArgumentParser(description='Prepare Billiard Data')
    parser.add_argument('--split', required=True, help='name of pickle file', type=str)
    return parser.parse_args()


def main():
    args = arg_parse()
    r = 2.0
    num_objs = 3
    name = 'simb_split'
    split = args.split
    print(split)
    tar_dir = f'data/{name}/{split}/'
    os.makedirs(tar_dir, exist_ok=True)
    src_file = f'data/{name}/{split}.hkl'
    src_data = hickle.load(src_file)
    image_data = src_data['X']
    label_data = src_data['y']
    mask_data = None
    env_meta = {}
    if src_data['board_mask'] is not None:
        mask_data = src_data['board_mask']
    if src_data['board_size'] is not None:
        env_meta.update({'board_size': src_data['board_size']})
    if 'split_point' in src_data.keys() and src_data['split_point'] is not None:
        env_meta.update({'split_point': src_data['split_point']})
    env_meta_dir = os.path.join(tar_dir, 'env_meta.pkl')
    with open(env_meta_dir, 'wb') as f:
        pickle.dump(env_meta, f, pickle.HIGHEST_PROTOCOL)
    num_seq, im_seq_len, im_h, im_w, num_channel = image_data.shape
    _, label_seq_len, _, _ = label_data.shape
    r = src_data['r'] if 'r' in src_data else r * np.ones((num_seq, num_objs))
    for i in range(num_seq):
        print(f'{i}/{num_seq}') if i % 100 == 0 else print('', end='')
        # 1. write out images
        cur_image_data = image_data[i]
        cur_label_data = label_data[i]
        cur_target_dir = os.path.join(tar_dir, f'{i:05d}')
        os.makedirs(cur_target_dir, exist_ok=True)
        cur_target_roi_name = os.path.join(tar_dir, f'{i:05d}.pkl')
        # 2. convert position to bounding box locations
        all_rois = np.zeros((label_seq_len, cur_label_data.shape[1], 5))
        for t in range(im_seq_len):
            cur_image = cur_image_data[t]
            with open(os.path.join(cur_target_dir, f'{t:05d}_data.pkl'), 'wb') as f:
                pickle.dump(cur_image * 255, f, pickle.HIGHEST_PROTOCOL)
            if mask_data is not None:
                mask = mask_data[i][t]
                with open(os.path.join(cur_target_dir, f'{t:05d}_bmask.pkl'), 'wb') as f:
                    pickle.dump(mask, f, pickle.HIGHEST_PROTOCOL)
            write_img = cur_image * 255
            write_img = Image.fromarray(write_img.astype(np.uint8) )
            write_img.save(os.path.join(cur_target_dir, f'{t:05d}_debug.png'))
            # cv2.imwrite(os.path.join(cur_target_dir, f'{t:05d}_debug.png'), cur_image * 255)
        for t in range(label_seq_len):
            cur_rois = cur_label_data[t]
            all_rois[t, :, 0] = np.arange(num_objs)
            for obj_id in range(num_objs):
                all_rois[t, obj_id, 1] = cur_rois[obj_id, 1] - r[i, obj_id]
                all_rois[t, obj_id, 2] = cur_rois[obj_id, 0] - r[i, obj_id]
                all_rois[t, obj_id, 3] = cur_rois[obj_id, 1] + r[i, obj_id]
                all_rois[t, obj_id, 4] = cur_rois[obj_id, 0] + r[i, obj_id]
        with open(cur_target_roi_name, 'wb') as f:
            pickle.dump(all_rois, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
