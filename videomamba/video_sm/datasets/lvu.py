import os
import os
import io
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop, 
)
from .video_transforms import ConsistentVideoTransforms

class LVU(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, prefix='', split=',', mode='train', clip_len=8,
                frame_sample_rate=2, crop_size=224, short_side_size=256,
                new_height=256, new_width=340, keep_aspect_ratio=True,
                num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                args=None, trimmed=60, time_stride=16):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.trimmed = trimmed
        self.time_stride = time_stride
        self.num_problem = 0
        print(f"Use trimmed videos of {trimmed} seconds")
        print(f"Time stride: {time_stride} seconds")
        assert num_segment == 1
        if self.mode in ['train']:
            self.aug = True
            if args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        self.cleaned = pd.read_csv(self.anno_path, delimiter=self.split)
        self.video_transform = ConsistentVideoTransforms(mode=self.mode)

        

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 

            sample = self.cleaned.iloc[index]
            file_name = sample['video']
            start = sample['start_frame']
            end = sample['stop_frame']
            label = sample['class_id']
            # print(file_name, start, end)
            buffer = self.loadvideo_decord(file_name, start, end, chunk_nb=-1) # T H W C
            if len(buffer) == 0:
                self.num_problem += 1
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.cleaned.iloc[index]
                    file_name = sample['video']
                    start = sample['start_frame']
                    end = sample['stop_frame']
                    label = sample['class_id']
                    buffer = self.loadvideo_decord(file_name, start, end, chunk_nb=0)
            buffer = self.video_transform(buffer)
            # print(buffer.shape)
            # exit(0)
            channel_dim = buffer.shape[1]
            if channel_dim < 16:
                # Determine the number of channels to repeat
                num_repeat = 16 - channel_dim
                
                # Repeat the last channel to pad the tensor
                buffer = torch.cat([buffer, buffer[:, -1:].repeat(1, num_repeat, 1, 1)], dim=1)

            return buffer, label, index, {}

        elif self.mode == 'validation':
            sample = self.cleaned.iloc[index]
            file_name = sample['video']
            start = sample['start_frame']
            end = sample['stop_frame']
            label = sample['class_id']
            buffer = self.loadvideo_decord(file_name, start, end, chunk_nb=0)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.cleaned.iloc[index]
                    file_name = sample['video']
                    start = sample['start_frame']
                    end = sample['stop_frame']
                    label = sample['class_id']
                    buffer = self.loadvideo_decord(file_name, start, end, chunk_nb=0)
            buffer = self.video_transform(buffer)
            
            channel_dim = buffer.shape[1]
            if channel_dim < 16:
                # Determine the number of channels to repeat
                num_repeat = 16 - channel_dim
                
                # Repeat the last channel to pad the tensor
                buffer = torch.cat([buffer, buffer[:, -1:].repeat(1, num_repeat, 1, 1)], dim=1)
            
            return buffer, label, index
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def loadvideo_decord(self, sample, start, end, chunk_nb=0):
        """Load video content using Decord"""
        fname = sample + ".masked.mp4"
        fname = os.path.join(self.prefix, fname)

        try:
            if self.keep_aspect_ratio:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                    num_threads=1,
                                    ctx=cpu(0))
                else:
                    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                if "s3://" in fname:
                    video_bytes = self.client.get(fname)
                    vr = VideoReader(io.BytesIO(video_bytes),
                                    width=self.new_width,
                                    height=self.new_height,
                                    num_threads=1,
                                    ctx=cpu(0))
                else:
                    vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                    num_threads=1, ctx=cpu(0))

            fps = vr.get_avg_fps()
            vr.seek(0)
            # print(list(range(start, end + 1)))
            batch_images = vr.get_batch(list(range(start, end + 1))).asnumpy()
            buffer = []
            for i, img in enumerate(batch_images):
                buffer.append(Image.fromarray(img))
            return buffer
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.cleaned) // 8
            # return 64
        else:
            return len(self.cleaned) // 8
            # return 64


if __name__ == "__main__":
    dataset = LVU(anno_path = 'data/train_refine.csv',prefix='/home/sobhan/Documents/Datasets/MASKED_VIDEOS/')
    buffer, label, *_ = dataset[0]
    print(label)
    for frames in buffer:
        print(frames.shape)
    print('*'*20)
    buffer, label, *_ = dataset[1]
    print(label)
    for frames in buffer:
        print(frames.shape)
    print('*'*20)
    buffer, label, *_ = dataset[2]
    print(label)
    for frames in buffer:
        print(frames.shape)
    print('*'*20)