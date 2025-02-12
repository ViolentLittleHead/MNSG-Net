import torch

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



