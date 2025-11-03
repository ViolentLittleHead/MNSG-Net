import numpy as np
import torch
from albumentations import Resize
from PIL import Image
from MNSGNet import MNSGNet
from torchvision.transforms.functional import to_pil_image, to_tensor
import util
from metric import calculate_iou, calculate_f1,calculate_hd95
import os
import glob


def createSample(transform, imagePath, maskPath):
    image = Image.open(imagePath)
    mask = Image.open(maskPath)

    image = np.array(image)
    mask = np.array(mask)

    if transform:
        augmented = transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
    image = to_tensor(image)
    # 规范化，将掩码缩小到【0，255】的范围
    normalized_mask = mask / 255.0
    mask = to_tensor(normalized_mask)
    return image, mask


if __name__ == "__main__":
    h, w = 256, 256
    dataset_name = "SuZhou"

    # 设置路径
    test_image_dir = f"../{dataset_name}/{dataset_name}/test/images/"
    test_mask_dir = f"../{dataset_name}/{dataset_name}/test/masks/"
    # test_image_dir = f"../SuZhou/para_SuZhou/train/images/"
    # test_mask_dir = f"../SuZhou/para_SuZhou/train/masks/"

    result_dir = f"./result/{dataset_name}"

    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 获取所有测试图像文件
    image_files = glob.glob(os.path.join(test_image_dir, "*.png"))

    # 创建模型
    model = MNSGNet(1, 1, img_size=h)
    model = model.to('cuda')

    path2weights = f"./models/{dataset_name}/weights20250530_09_39_29.pt"
    model.load_state_dict(torch.load(path2weights))
    model.eval()

    # 初始化指标列表
    dice_scores = []
    iou_scores = []
    f1_scores = []
    hd95_scores = []
    # 创建变换
    transform = Resize(h, w)

    # 处理每个图像
    for image_path in image_files:
        # 获取文件名
        file_name = os.path.basename(image_path)

        # 构建掩码路径 - 假设掩码文件名与图像文件名相同
        mask_path = os.path.join(test_mask_dir, file_name)

        # 检查掩码文件是否存在
        if not os.path.exists(mask_path):
            print(f"Warning: not find MASK {mask_path}，跳过此图像")
            continue

        # 创建样本
        xb, yb = createSample(transform, image_path, mask_path)

        # 添加批次维度 [C, H, W] -> [1, C, H, W]
        xb = xb.unsqueeze(0).to('cuda')
        yb = yb.unsqueeze(0).to('cuda')


        # 预测
        with torch.no_grad():
            output = model(xb)
            pred = torch.sigmoid(output)

            # 计算指标
            _, dice_score = util.dice_loss(pred, yb)
            iou_score = calculate_iou(pred, yb)
            f1_score = calculate_f1(pred, yb)

            # 计算HD95
            prediction_binary = (pred > 0.5).float()
            hd95 = calculate_hd95(prediction_binary, yb)

            # 记录指标
            dice_scores.append(dice_score.item())
            iou_scores.append(iou_score.item())
            f1_scores.append(f1_score.item())
            hd95_scores.append(hd95)  # 添加HD95分数

           # 打印当前样本的指标值
            print(f"image: {file_name} - Dice={dice_score:.4f}, IoU={iou_score:.4f}, F1={f1_score:.4f}, HD95={hd95:.4f}")

            # 二值化预测结果
            prediction_binary = (pred > 0.5).float()

            # 保存预测结果
            # 移除批次维度和通道维度 [1, 1, H, W] -> [H, W]
            pred_image = to_pil_image(prediction_binary.squeeze(0).squeeze(0).cpu())

            # 构建保存路径
            save_path = os.path.join(result_dir, file_name)

            # 保存图像
            pred_image.save(save_path)
            print(f"Save to Path: {save_path}")

     # 计算并打印平均指标
    if dice_scores:
        avg_dice = sum(dice_scores) / len(dice_scores)
        avg_iou = sum(iou_scores) / len(iou_scores)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_hd95 = sum(hd95_scores) / len(hd95_scores)  # 计算平均HD95
        
        print("\n===== Metrics =====")
        print(f"Avg Dice : {avg_dice:.4f}")
        print(f"Avg IoU  : {avg_iou:.4f}")
        print(f"Avg F1   : {avg_f1:.4f}")
        print(f"Avg HD95 : {avg_hd95:.4f}")  # 打印平均HD95
        print(f"Total Number: {len(dice_scores)}")
    else:
        print("ERROR!")