import os
import random
from itertools import repeat
from multiprocessing.pool import Pool
import contextlib
import numpy as np
import torch
from torch.utils.data import DataLoader, distributed
from pathlib import Path
from ..augmentations import augment_hsv, copy_paste, letterbox
from ..dataloaders import InfiniteDataLoader,LoadImagesAndLabels,seed_worker,TQDM_BAR_FORMAT,get_hash,HELP_URL
from ..general import LOGGER, xyn2xy, NUM_THREADS
from ..torch_utils import torch_distributed_zero_first
from .augmentations import mixup, random_perspective
from tqdm import tqdm
from PIL import ExifTags, Image, ImageOps
from utils.obb.utils import poly2rbox,xyxyxyxy2xywhr
from .plots import imglabvisualize, imglabvisualize2
import cv2

RANK = int(os.getenv('RANK', -1))
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'

def create_dataloader(path,
                      imgsz,
                      batch_size,
                      stride,
                      single_cls=False,
                      hyp=None,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      rect=False,
                      rank=-1,
                      workers=8,
                      image_weights=False,
                      close_mosaic=False,
                      quad=False,
                      prefix='',
                      shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadObbImagesAndLabels(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augmentation
            hyp=hyp,  # hyperparameters
            rect=rect,  # rectangular batches
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    #loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    loader = DataLoader if image_weights or close_mosaic else InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadObbImagesAndLabels.collate_fn4 if quad else LoadObbImagesAndLabels.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset

class LoadObbImagesAndLabels(LoadImagesAndLabels):  # for training/testing

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix=""
    ):
        super().__init__(path, img_size, batch_size, augment, hyp, rect, image_weights, cache_images, single_cls,
                         stride, pad, min_items, prefix)


    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        # masks = []
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:

                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
                # cv2.imshow("",img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # imglabvisualize(img,labels)
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            # [array, array, ....], array.shape=(num_points, 2), xyxyxyxy
            # segments = self.segments[index].copy()
            # if len(segments):
            #     for i_s in range(len(segments)):
            #         segments[i_s] = xyn2xy(
            #             segments[i_s],
            #             ratio[0] * w,
            #             ratio[1] * h,
            #             padw=pad[0],
            #             padh=pad[1],
            #         )
            # # 将标签坐标从(x, y, w, h)的格式转换为(x1, y1, x2, y2)的格式
            if labels.size:  # normalized xywh to pixel xyxy format
                # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                labels[:, [1, 3, 5, 7]] = self.labels[index][:, [1, 3, 5, 7]] * ratio[0] * w + pad[0]
                labels[:, [2, 4, 6, 8]] = self.labels[index][:, [2, 4, 6, 8]] * ratio[1] * h + pad[1]
            if self.augment:
                img, labels = random_perspective(img,labels,
                                               degrees=hyp["degrees"],
                                               translate=hyp["translate"],
                                               scale=hyp["scale"],
                                               shear=hyp["shear"],
                                               perspective=hyp["perspective"])
                # imglabvisualize(img, labels,name="152")
        nl = len(labels)  # number of labels
        # if nl:
            # 将标签坐标从(x1, y1, x2, y2)的格式转换为(x, y, w, h)的格式，并规范化到[0, 1]的范围内
            # labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
            # if self.overlap:
            #     masks, sorted_idx = polygons2masks_overlap(img.shape[:2],
            #                                                segments,
            #                                                downsample_ratio=self.downsample_ratio)
            #     masks = masks[None]  # (640, 640) -> (1, 640, 640)
            #     labels = labels[sorted_idx]
            # else:
            #     masks = polygons2masks(img.shape[:2], segments, color=1, downsample_ratio=self.downsample_ratio)

        # masks = (torch.from_numpy(masks) if len(masks) else torch.zeros(1 if self.overlap else nl, img.shape[0] //
        #                                                                 self.downsample_ratio, img.shape[1] //
        #                                                                 self.downsample_ratio))
        # TODO: albumentations support
        if self.augment:
            # Albumentations
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            # img, labels = self.albumentations(img, labels)
            # nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    # labels[:, 2] = 1 - labels[:, 2]
                    labels[:, 2::2] = img.shape[0] - labels[:, 2::2] - 1
                    # masks = torch.flip(masks, dims=[1])

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    # labels[:, 1] = 1 - labels[:, 1]
                    labels[:, 1::2] = img.shape[1] - labels[:, 1::2] - 1

        if nl:
            # rboxes = poly2rbox(polys=labels[:, 1:9],
            #                                 num_cls_thata= 180,  # cls_theta
            #                                 radius = 6.0, # csl_radius
            #                                 use_pi=True, use_gaussian=False)
            rboxes = xyxyxyxy2xywhr(labels[:,1:9])
            labels_obb = np.concatenate((labels[:, :1], rboxes), axis=1)
            labels_mask = (rboxes[:, 0] >= 0) & (rboxes[:, 0] < img.shape[1]) \
                          & (rboxes[:, 1] >= 0) & (rboxes[:, 0] < img.shape[0]) \
                          & (rboxes[:, 2] > 5) | (rboxes[:, 3] > 5)     # 过滤
            labels_obb = labels_obb[labels_mask]
            nl = len(labels_obb)  # update after filter
        # imglabvisualize(img, labels_obb[:,1:], name=index)
        labels_out = torch.zeros((nl, 7))
        if nl:
            # labels_out[:, 1:] = torch.from_numpy(labels)
            labels_out[:, 1:] = torch.from_numpy(labels_obb)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return (torch.from_numpy(img), labels_out, self.im_files[index], shapes)

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                # 将归一化的坐标转为绝对坐标
                # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
                labels[:, [1, 3, 5, 7]] = self.labels[index][:, [1, 3, 5, 7]]*w + padw
                labels[:, [2, 4, 6, 8]] = self.labels[index][:, [2, 4, 6, 8]]*h + padh

                segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
            labels4.append(labels)
            segments4.extend(segments)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        # for x in (labels4[:, 1:], *segments4):
        for x in (segments4):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        h_filter = 2 * s
        w_filter = 2 * s
        labels_mask = poly_filter(polys=labels4[:, 1:].copy(), h=h_filter, w=w_filter)
        labels4 = labels4[labels_mask]
        # img4, labels4 = replicate(img4, labels4)  # replicate
        # imglabvisualize(img4,labels4)
        # Augment
        img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           segments4,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4


    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    def cache_labels(self, path = Path('./labels.cache'), prefix = ''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                        desc = desc,
                        total = len(self.im_files),
                        bar_format = TQDM_BAR_FORMAT)
            for im_file, lb, shape,segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 9, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 9), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 9), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def poly_filter(polys, h, w):
    """
    Filter the poly labels which is out of the image.
    Args:
        polys (array): (num, 8)

    Return：
        keep_masks (array): (num)
    """
    x = polys[:, 0::2] # (num, 4)
    y = polys[:, 1::2]
    x_max = np.amax(x, axis=1) # (num)
    x_min = np.amin(x, axis=1)
    y_max = np.amax(y, axis=1)
    y_min = np.amin(y, axis=1)
    x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
    keep_masks = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h)
    return keep_masks