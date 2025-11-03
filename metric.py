import torch
from skimage import measure
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(a, b, distance='euclidean'):
    """计算双向 Hausdorff 距离"""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

# 定义HD95计算函数
def calculate_hd95(pred, target):
    """
    批量计算HD95距离
    :param pred: 预测张量 [B, C, H, W]
    :param target: 目标张量 [B, C, H, W]
    :return: 平均HD95值 (float)
    """

    def get_surface_points(mask):
        mask_np = np.squeeze(mask.cpu().numpy())
        if mask_np.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask_np.shape}")
        if mask_np.max() == 0:  # 无前景区域
            return np.zeros((0, 2))
        contours = measure.find_contours(mask_np, 0.5)
        points = np.concatenate([c for c in contours]) if len(contours) > 0 else np.zeros((0, 2))
        return points

    pred = pred.detach().cpu()
    target = target.detach().cpu()

    B, C = pred.shape[:2]
    hd_values = []

    for b in range(B):
        for c in range(C):
            pred_mask = pred[b, c]
            target_mask = target[b, c]

            pred_points = get_surface_points(pred_mask)
            target_points = get_surface_points(target_mask)

            # 处理空表面情况
            if len(pred_points) == 0 and len(target_points) == 0:
                hd = 0.0
            elif len(pred_points) == 0 or len(target_points) == 0:
                h, w = pred_mask.shape[-2:]
                hd = np.sqrt(h ** 2 + w ** 2)
            else:
                hd = hausdorff_distance(pred_points, target_points, distance='euclidean')

            hd_values.append(hd)

    # 返回平均HD95
    return float(np.sum(hd_values))

def calculate_iou(pred, target, smooth=1e-5):
    # 二值化预测值 (通过阈值 0.5)
    pred_binary = (pred > 0.5).float()  # 将预测值二值化，大于0.5为1，其他为0

    # 计算交集（Intersection）和并集（Union）
    intersection = (pred_binary * target).sum(dim=(2, 3))  # 计算交集：预测和目标都为1的像素数
    union = pred_binary.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection  # 计算并集：预测为1或目标为1的像素数

    # 计算 IoU
    iou = (intersection + smooth) / (union + smooth)  # 加上平滑项，避免除以零

    # 返回 IoU 总和和每个样本的 IoU 值
    return iou.sum()


def calculate_f1(pred, target, threshold=0.5, smooth=1e-5):
    # 将预测值二值化 (大于threshold为1，否则为0)
    pred_binary = (pred > threshold).float()  # 二值化预测值
    
    # 计算 TP, FP, FN
    TP = (pred_binary * target).sum(dim=(2, 3))  # 预测为1且真实为1的像素数
    FP = ((pred_binary == 1) & (target == 0)).sum(dim=(2, 3))  # 预测为1且真实为0的像素数
    FN = ((pred_binary == 0) & (target == 1)).sum(dim=(2, 3))  # 预测为0且真实为1的像素数
    
    # 计算 Precision 和 Recall
    precision = TP / (TP + FP + smooth)  # 精确度
    recall = TP / (TP + FN + smooth)  # 召回率

    # 计算 F1 系数
    f1 = 2 * (precision * recall) / (precision + recall + smooth)  # 加平滑项避免除以零

    return f1.sum()



