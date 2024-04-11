import numpy as np
import torch
import cv2
import time
import math
from utils.obb.loss_tal import _get_covariance_matrix
pi = 3.141592


def xyxyxyxy2xywhr(corners):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation]. Rotation values are
    expected in degrees from 0 to 90.

    Args:
        corners (numpy.ndarray | torch.Tensor): Input corners of shape (n, 8).

    Returns:
        (numpy.ndarray | torch.Tensor): Converted data in [cx, cy, w, h, rotation] format of shape (n, 5).
    """
    is_torch = isinstance(corners, torch.Tensor)
    points = corners.cpu().numpy() if is_torch else corners
    points = points.reshape(len(corners), -1, 2)
    rboxes = []
    for pts in points:
        # NOTE: Use cv2.minAreaRect to get accurate xywhr,
        # especially some objects are cut off by augmentations in dataloader.
        (x, y), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([x, y, w, h, angle / 180 * np.pi])
    rboxes = (
        torch.tensor(rboxes, device=corners.device, dtype=corners.dtype)
        if is_torch
        else np.asarray(rboxes, dtype=points.dtype)
    )
    return rboxes



def poly2rbox(polys, num_cls_thata=180, radius=6.0, use_pi=False, use_gaussian=False):
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
    if use_gaussian:
        csl_labels = []
    rboxes = []
    # print(polys)
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        angle = -angle # θ ∈ [-90， 0]
        theta = angle / 180 * pi # 转为pi制
        # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
        if w != max(w, h):
            w, h = h, w
            theta += pi/2

        # theta = regular_theta(theta) # limit theta ∈ [-pi/2, pi/2)

        angle = (theta * 180 / pi) + 90 # θ ∈ [0， 180)

        # #x,y,w,h->x1,y1,x2,y2
        # x1=x-w/2
        # y1=x-h/2
        # x2=x+w/2
        # y2=y+h/2

        if not use_pi: # 采用angle弧度制 θ ∈ [0， 180)
            rboxes.append([x, y, w, h, angle])
        else: # 采用pi制
            rboxes.append([x, y, w, h, theta])
            # rboxes.append([x1, y1, x2, y2, theta])
        if use_gaussian:
            csl_label = gaussian_label_cpu(label=angle, num_class=num_cls_thata, u=0, sig=radius)
            csl_labels.append(csl_label)
    if use_gaussian:
        return np.array(rboxes), np.array(csl_labels)
    return np.array(rboxes)

def gaussian_label_cpu(label, num_class, u=0, sig=4.0):
    """
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], μ in gaussian function
        sig (float32):[1], σ in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class/2, num_class/2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class/2 - label)
    return np.concatenate([y_sig[index:],
                           y_sig[:index]], axis=0)


def rotated_non_max_suppression(prediction, conf_thres=0.45, iou_thres=0.75, classes=None, agnostic=False,
                            multi_label=False,
                            labels=(), max_det=1500):
    """Runs Non-Maximum Suppression (NMS) on inference results_obb
    Args:
        prediction (tensor): (b, n_all_anchors, [cx cy l s obj num_cls theta_cls])
        agnostic (bool): True = NMS will be applied between elements of different categories
        labels : () or

    Returns:
        list of detections, len=batch_size, on (n,7) tensor per image [xylsθ, conf, cls] θ ∈ [-pi/2, pi/2)
    """

    # conf_thres=0.3
    # iou_thres=0.1

    if isinstance(prediction, (list, tuple)):  # YOLO model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - 0 - 5  # number of classes
    # nc = prediction.shape[2] - 5   # number of classes
    # xc = prediction[..., 5] > conf_thres  # candidates
    mi = 5 + nc  # mask start index
    xc = prediction[:, 5:mi].amax(1) > conf_thres  # candidates

    # Settings
    max_wh = 7680  # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 30.0  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)


    t = time.time()
    output = [torch.zeros((0, 7), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        box, cls = x.split((5, nc), 1)

        theta_pred = x[:, 4]
        theta_pred = (theta_pred.sigmoid() - 0.5) * math.pi  # [n_conf_thres, 1] θ ∈ [-pi/2, pi/2)
        theta_pred = theta_pred[:, np.newaxis]

        # Detections matrix nx7 (xyls, θ, conf, cls) θ ∈ [-pi/2, pi/2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T  # ()
            x = torch.cat((x[i, :4], theta_pred[i], x[i, j + 5, None], j[:, None].float()), 1)

        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((x[:, :4], theta_pred, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            # x = torch.cat((x[:, :5], conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # elif n > max_nms:  # excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        rboxes = x[:, :5].clone()
        rboxes[:, :2] = rboxes[:, :2] + c  # rboxes (offset by class)
        scores = x[:, 5]

        # _, i = obb_nms(rboxes, scores, iou_thres)  #dets (tensor/array): (num, [cx cy w h θ]) θ∈[-pi/2, pi/2)
        i = nms_rotated(rboxes, scores, iou_thres)

        # if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]

        output[xi] = x[i]
    # 输出nms之后的旋转框和对应的关键点
    return output
def nms_rotated(boxes, scores, threshold=0.45):
    """
    NMS for obbs, powered by probiou and fast-nms.

    Args:
        boxes (torch.Tensor): (N, 5), xywhr.
        scores (torch.Tensor): (N, ).
        threshold (float): Iou threshold.

    Returns:
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    return 1 - hd