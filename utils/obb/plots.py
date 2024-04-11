import contextlib
import math
from pathlib import Path
import seaborn as sn
import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from .. import threaded
from ..general import xywh2xyxy,LOGGER
from ..plots import Annotator, colors
pi = 3.141592

@threaded
def plot_obb_labels(labels, names=(), save_dir=Path(''), img_size=1024):
    rboxes = poly2rbox(labels[:, 1:])
    labels = np.concatenate((labels[:, :1], rboxes[:, :-1]), axis=1) # [cls xyls]

    # plot dataset labels
    # LOGGER.info(f"Plotting labels to {save_dir / 'labels_xyls.jpg'}... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, hboxes(xyls)
    nc = int(c.max() + 1)  # number of classes
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'long_edge', 'short_edge'])

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    # [y[2].patches[i].set_color([x / 255 for x in colors(i)]) for i in range(nc)]  # update colors bug #3195
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='long_edge', y='short_edge', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    # labels[:, 1:3] = 0.5 # center
    labels[:, 1:3] = 0.5 * img_size # center
    # labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    labels[:, 1:] = xywh2xyxy(labels[:, 1:])
    # img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    img = Image.fromarray(np.ones((img_size, img_size, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels_xyls.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

def poly2rbox(polys,use_pi=False):
    """
    Trans poly format to rbox format.
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
        num_cls_thata (int): [1], theta class num
        radius (float32): [1], window radius for Circular Smooth Label
        use_pi (bool): True θ∈[-pi/2, pi/2) ， False θ∈[0, 180)

    Returns:
        use_gaussian True:
            rboxes (array):
            csl_labels (array): (num_gts, num_cls_thata)
        elif
            rboxes (array): (num_gts, [cx cy l s θ])
    """
    assert polys.shape[-1] == 8
    rboxes = []
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        angle = -angle # θ ∈ [-90， 0]
        theta = angle / 180 * pi # 转为pi制

        # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
        if w != max(w, h):
            w, h = h, w
            theta += pi/2
        theta = regular_theta(theta) # limit theta ∈ [-pi/2, pi/2)
        angle = (theta * 180 / pi) + 90 # θ ∈ [0， 180)

        if not use_pi: # 采用angle弧度制 θ ∈ [0， 180)
            rboxes.append([x, y, w, h, angle])
        else: # 采用pi制
            rboxes.append([x, y, w, h, theta])
    return np.array(rboxes)

def regular_theta(theta, mode='180', start=-pi/2):
    """
    limit theta ∈ [-pi/2, pi/2)
    """
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def imglabvisualize(img, label_,name):
    # 可视化obb的标签
    img = np.ascontiguousarray(img)
    img_ = np.transpose(img, (1,2,0)).astype(np.uint8).copy()
    # img_ = np.transpose(img, (2, 0, 1)).astype(np.uint8).copy()
    for rect_info in label_:
        x, y, w, h, angle_rad = rect_info
        rect_center = (int(x), int(y))
        rect_size = (int(w), int(h))
        rect_angle = math.degrees(angle_rad)  # 将弧度转换为角度制
        box = (rect_center, rect_size, rect_angle)
        pts = cv2.boxPoints(box).astype(np.int0)  # 计算矩形的顶点坐标
        cv2.polylines(img_, [pts], isClosed=True, color=(255, 0, 0), thickness=2)  # 在图像上绘制矩形
    # cv2.imshow('Image with rectangles', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r"E:/study/yolov9-rotated/yolov9-rotated/runs/train-obb/img/"+str(name)+"_r.jpg", img_)

def imglabvisualize2(img,label_,name):
    # 可视化obb的标签
    img = np.ascontiguousarray(img)
    # m = np.transpose(img,(1, 2, 0)).astype(np.uint8).copy()
    for rect in label_:
        x1, y1, x2, y2, x3, y3, x4, y4 = rect[0], rect[1], rect[2], rect[3], rect[4], rect[5], rect[6], rect[
            7]
        points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)  # 重组顶点坐标
        points = points.reshape((-1, 1, 2))
        color = (128, 128, 128)     # if category == 1 else (0, 0, 255)  # 根据类别选择颜色
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)  # 在图像上绘制多边形

    cv2.imwrite(r"E:/study/yolov9-rotated/yolov9-rotated/runs/train-obb/img/"+str(name)+"_4.jpg", img)



def poly2hbb(polys):
    """
    Trans poly format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, poly)

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h])
    """
    assert polys.shape[-1] == 8
    if isinstance(polys, torch.Tensor):
        x = polys[:, 0::2] # (num, 4)
        y = polys[:, 1::2]
        x_max = torch.amax(x, dim=1) # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = torch.cat((x_ctr, y_ctr, w, h), dim=1)
    else:
        x = polys[:, 0::2] # (num, 4)
        y = polys[:, 1::2]
        x_max = np.amax(x, axis=1) # (num)
        x_min = np.amin(x, axis=1)
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = np.concatenate((x_ctr, y_ctr, w, h), axis=1)
    return hbboxes

def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]

        Cos, Sin = torch.cos(theta), torch.sin(theta)


        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)

def scale_polys(img1_shape, polys, img0_shape, ratio_pad=None):
    # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0] # h_ratios
        pad = ratio_pad[1] # wh_paddings

    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain # Rescale poly shape to img0_shape
    #clip_polys(polys, img0_shape)
    return polys

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
@threaded
def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=2048, max_subplots=4):
    """
    Args:
        imgs (tensor): (b, 3, height, width)
        targets_train (tensor): (n_targets, [batch_id clsid cx cy l s theta gaussian_θ_labels]) θ∈[-pi/2, pi/2)
        targets_pred (array): (n, [batch_id, class_id, cx, cy, l, s, theta, conf]) θ∈[-pi/2, pi/2)
        paths (list[str,...]): (b)
        fname (str): (1)
        names :

    """
    # Plot image grid with labels
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y:y + h, x:x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # image targets, (n, [img_index clsid cx cy l s theta gaussian_θ_labels])
            # boxes = xywh2xyxy(ti[:, 2:6]).T
            rboxes = ti[:, 2:7]
            classes = ti[:, 1].astype('int')
            # labels = ti.shape[1] == 6  # labels if no conf column
            labels = ti.shape[1] == 187  # labels if no conf column
            # conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)
            conf = None if labels else ti[:, 7]  # check for confidence presence (label vs pred)

            # if boxes.shape[1]:
            #     if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
            #         boxes[[0, 2]] *= w  # scale to pixels
            #         boxes[[1, 3]] *= h
            #     elif scale < 1:  # absolute coords need scale if image scales
            #         boxes *= scale
            polys = rbox2poly(rboxes)
            if scale < 1:
                polys *= scale
            # boxes[[0, 2]] += x
            # boxes[[1, 3]] += y
            polys[:, [0, 2, 4, 6]] += x
            polys[:, [1, 3, 5, 7]] += y
            # for j, box in enumerate(boxes.T.tolist()):
            #     cls = classes[j]
            #     color = colors(cls)
            #     cls = names[cls] if names else cls
            #     if labels or conf[j] > 0.25:  # 0.25 conf thresh
            #         label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
            #         annotator.box_label(box, label, color=color)
            for j, poly in enumerate(polys.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    annotator.box_label(poly, label, color=color)
    annotator.im.save(fname)  # save


def output_to_target(output): #list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
    # Convert model output to target format [batch_id, class_id, x, y, l, s, theta, conf]
    targets = []
    for i, o in enumerate(output):
        for *rbox, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*(np.array(rbox)[None])), conf])
    return np.array(targets)
