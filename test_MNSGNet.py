import numpy as np
import torch
from albumentations import Resize
import util
from MNSGNet import MNSGNet
from fetaL_dataset import fetaL_dataset
from torch.utils.data import DataLoader
from metric import calculate_iou, calculate_f1, calculate_hd95

h, w = 256, 256
dataset_name = "OSF"
trainform_test = Resize(h, w)
path2train = "../../" + dataset_name + "/test"
test_ds = fetaL_dataset(path2train, transform=trainform_test)
test_dl = DataLoader(test_ds, batch_size=4, shuffle=False)

"""
    创建模型
"""
model = MNSGNet(1, 1, img_size=h)
model = model.to('cuda')

path2weights = "./models/" + dataset_name + "/weights20250530_10_15_36.pt"
model.load_state_dict(torch.load(path2weights))
model.eval()

def print_stats(name, values):
    """打印统计信息：平均值、最大值、最小值"""
    print(f"\n{name} - AVG: {values.mean():.4f}, MAX: {values.max():.4f}, MIN: {values.min():.4f}")

len_data = len(test_dl.dataset)
all_dice = []  # 存储每个样本的Dice分数
all_iou = []   # 存储每个样本的IoU分数
all_f1 = []    # 存储每个样本的F1分数
all_hd = []    # 存储每个样本的hd95


print("Start：")
print("="*60)

i = 0
for batch_idx, (xb, yb) in enumerate(test_dl):
    xb = xb.to('cuda')
    yb = yb.to('cuda')

    output = model(xb)
    with torch.no_grad():
        pred = torch.sigmoid(output)
        
        # 计算当前batch中每个样本的指标
        for j in range(pred.shape[0]):
            sample_idx = batch_idx * test_dl.batch_size + j
            pred_sample = pred[j].unsqueeze(0)  # 保持维度 [1, 1, H, W]
            yb_sample = yb[j].unsqueeze(0)
            
            # 计算单个样本的指标
            _, dice_score = util.dice_loss(pred_sample, yb_sample)
            iou_score = calculate_iou(pred_sample, yb_sample)
            f1_score = calculate_f1(pred_sample, yb_sample)
            hd_score = calculate_hd95(pred_sample, yb_sample)
            
            # 存储指标
            dice_val = dice_score.item()
            iou_val = iou_score.item()
            f1_val = f1_score.item()
            hd_val = hd_score.item()
            
            all_dice.append(dice_val)
            all_iou.append(iou_val)
            all_f1.append(f1_val)
            all_hd.append(hd_val)
            
            # 打印当前样本的指标值
            print(f"样本 {sample_idx+1:03d}/{len_data}: " 
                  f"Dice={dice_val:.4f}, IoU={iou_val:.4f}, F1={f1_val:.4f}, HD={hd_val:.4f}")

# 转换为numpy数组便于计算
all_dice = np.array(all_dice)
all_iou = np.array(all_iou)
all_f1 = np.array(all_f1)

# 打印整体统计信息
print("\n" + "="*60)
print("最终统计结果:")
print_stats('Dice', all_dice)
print_stats('IoU', all_iou)
print_stats('F1', all_f1)
print_stats('HD', all_hd)

# 可选：打印其他统计信息
print("\n附加统计信息:")
print(f"总样本数: {len_data}")
print(f"Dice > 0.9 的样本数: {np.sum(all_dice > 0.9)}")
print(f"Dice < 0.1 的样本数: {np.sum(all_dice < 0.1)}")
print(f"Dice 中位数: {np.median(all_dice):.4f}")
print(f"Dice 标准差: {np.std(all_dice):.4f}")
