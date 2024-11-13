import os
import os
import io
import random
import numpy as np
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
from .volume_transforms import ClipToTensor

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

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

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = Compose([
                Resize(self.short_side_size, interpolation='bilinear'),
                CenterCrop(size=(self.crop_size, self.crop_size)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            self.test_start_array = []
            self.test_end_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_start_array.append(self.start_array[idx])
                        self.test_end_array.append(self.end_array[idx])
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

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
            # if args.num_sample > 1:
            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            print(buffer.shape)
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
            buffer = self.data_transform(buffer)
            
            channel_dim = buffer.shape[1]
            if channel_dim < 16:
                # Determine the number of channels to repeat
                num_repeat = 16 - channel_dim
                
                # Repeat the last channel to pad the tensor
                buffer = torch.cat([buffer, buffer[:, -1:].repeat(1, num_repeat, 1, 1)], dim=1)
            
            return buffer, label, index

        elif self.mode == 'test':
            sample = self.cleaned.iloc[index]
            file_name = sample['video']
            start = sample['start_frame']
            end = sample['stop_frame']
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample, start, end, chunk_nb=chunk_nb)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                start = self.test_start_array[index]
                end = self.test_end_array[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample, start, end, chunk_nb=chunk_nb)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)
            if self.test_num_crop == 1:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) / 2
                spatial_start = int(spatial_step)
            else:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                    / (self.test_num_crop - 1)
                spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[:, spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[:, :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                    chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer
    
    def _get_seq_frames(self, video_size, start, end, num_frames, clip_idx=-1):
        # 确保 start 和 end 在视频大小范围内
        start = max(0, min(start, video_size - 1))
        end = max(start, min(end, video_size - 1))

        # 计算裁剪后视频的实际大小
        clipped_video_size = end - start + 1
        seg_size = max(0., float(clipped_video_size - 1) / num_frames)
        seq = []

        if clip_idx == -1:
            for i in range(num_frames):
                start_frame = int(np.round(seg_size * i)) + start
                end_frame = int(np.round(seg_size * (i + 1))) + start
                idx = min(random.randint(start_frame, end_frame), end)
                seq.append(idx)
        else:
            num_segment = 1
            if self.mode == 'test':
                num_segment = self.test_num_segment
            duration = seg_size / (num_segment + 1)
            for i in range(num_frames):
                start_frame = int(np.round(seg_size * i)) + start
                frame_index = start_frame + int(duration * (clip_idx + 1))
                idx = min(frame_index, end)
                seq.append(idx)
        return seq

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
            buffer = vr.get_batch(list(range(start, end + 1))).asnumpy()
            return buffer
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

    def __len__(self):
        if self.mode != 'test':
            return len(self.cleaned)
        else:
            return len(self.cleaned)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = random_crop(frames, crop_size)
        else:
            transform_func = (
                random_resized_crop_with_shift
                if motion_shift
                else random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

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