from MNSGNet import MNSGNet
from fetaL_dataset import fetaL_dataset
import torch
import json
import os
import copy
from torch import optim
import util
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations import (HorizontalFlip, VerticalFlip, Compose, Resize, RandomRotate90)
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import logging

#============================ 参数配置区域 ============================#
# 数据集参数
DATASET_NAME = "OSF"
IMAGE_SIZE = (256, 256)          # 输入图像尺寸 (H, W)
TRAIN_DATA_PATH = "../../{}/train".format(DATASET_NAME)  # 训练数据路径
VAL_RATIO = 0.1                  # 验证集比例
RANDOM_SEED = 0                  # 随机种子

# 数据增强参数
AUG_PROB = {
    'horizontal_flip': 0.5,      # 水平翻转概率
    'vertical_flip': 0.1,        # 垂直翻转概率
    'rotate90': 0.5              # 90度旋转概率
}

# 训练参数
BATCH_SIZE = 10                  # 批大小
NUM_EPOCHS = 150                  # 总训练轮数
LEARNING_RATE = 1e-04          # 初始学习率
OPTIMIZER_PARAMS = {             # 优化器参数
    'type': 'Adam',               # 优化器类型
    'betas': (0.9, 0.999),        # Adam参数
    'weight_decay': 1e-4          # 权重衰减
}
LR_SCHEDULER_PARAMS = {          # 学习率调度器参数
    'mode': 'min',                # 监控指标模式
    'factor': 0.5,                # 学习率衰减因子
    'patience': 10,               # 等待周期数
    'verbose': True               # 是否显示信息
}
SANITY_CHECK = False             # 快速验证模式开关

# 模型参数
MODEL_PARAMS = {                 # 模型结构参数
    'in_ch': 1,             # 输入通道数
    'out_ch': 1,            # 输出通道数
    'img_size': IMAGE_SIZE[0]     # 输入图像尺寸
}
USE_PRETRAINED = True           # 是否使用预训练权重
PRETRAINED_PATH = './models/Spider/weights20250530_04_13_10.pt'  # 预训练路径

# 日志和保存参数
SAVE_ROOT = './'                 # 根保存路径
LOG_NAME_FORMAT = 'training{}.log'  # 日志文件名格式
MODEL_SAVE_FORMAT = 'weights{}.pt'  # 模型保存格式
LOG_INFO_SAVE_FORMAT = 'log_info{}.json'  # 训练信息保存格式
#=====================================================================#

# 自动生成路径
startDate = datetime.now().strftime("%Y%m%d_%H_%M_%S")
LOG_DIR = os.path.join(SAVE_ROOT, 'logs', DATASET_NAME)
MODEL_SAVE_DIR = os.path.join(SAVE_ROOT, 'models', DATASET_NAME)

# 初始化目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    filename=os.path.join(LOG_DIR, LOG_NAME_FORMAT.format(startDate)),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_serializable_config():
    """生成可序列化的配置字典"""
    return {
        'dataset': {
            'name': DATASET_NAME,
            'input_size': IMAGE_SIZE,
            'val_ratio': VAL_RATIO
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'epochs': NUM_EPOCHS,
            'initial_lr': LEARNING_RATE,
            'optimizer': {
                'type': OPTIMIZER_PARAMS['type'],
                'betas': list(OPTIMIZER_PARAMS['betas']),
                'weight_decay': OPTIMIZER_PARAMS['weight_decay']
            }
        },
        'model': {
            'architecture': 'MNSGNet',
            'params': MODEL_PARAMS
        }
    }

# 记录配置信息
logging.info("Training Configuration:\n%s",json.dumps(get_serializable_config(), indent=2))

#=========================== 数据准备 ===========================#
# 数据增强配置
transform_train = Compose([
    Resize(*IMAGE_SIZE),
    HorizontalFlip(p=AUG_PROB['horizontal_flip']),
    VerticalFlip(p=AUG_PROB['vertical_flip']),
    RandomRotate90(p=AUG_PROB['rotate90'])
], is_check_shapes=False)

transform_val = Resize(*IMAGE_SIZE)

# 加载数据集
fetal_ds_train = fetaL_dataset(TRAIN_DATA_PATH, transform=transform_train)
fetal_ds_val = fetaL_dataset(TRAIN_DATA_PATH, transform=transform_val)

# 划分验证集
sss = ShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=RANDOM_SEED)
indices = range(len(fetal_ds_train))
for train_index, val_index in sss.split(indices):
    logging.info(f'Train samples: {len(train_index)}, Val samples: {len(val_index)}')

# 创建数据加载器
train_ds = Subset(fetal_ds_train, train_index)
val_ds = Subset(fetal_ds_val, val_index)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

#=========================== 模型初始化 ===========================#
model = MNSGNet(**MODEL_PARAMS).cuda()
if USE_PRETRAINED and os.path.exists(PRETRAINED_PATH):
    model.load_state_dict(torch.load(PRETRAINED_PATH))
    logging.info("Loaded pretrained weights from: %s", PRETRAINED_PATH)

#=========================== 优化器配置 ===========================#
optimizer = getattr(optim, OPTIMIZER_PARAMS['type'])(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=OPTIMIZER_PARAMS['betas'],
    weight_decay=OPTIMIZER_PARAMS['weight_decay']
)
lr_scheduler = ReduceLROnPlateau(optimizer, **LR_SCHEDULER_PARAMS)

#=========================== 训练配置 ===========================#
params_train = {
    "num_epochs": NUM_EPOCHS,
    "optimizer": optimizer,
    "loss_func": util.loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": SANITY_CHECK,
    "lr_scheduler": lr_scheduler,
    "path2weights": os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_FORMAT.format(startDate))
}

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
"""
    计算每个epoch的loss和metric
"""
def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_dice=0.0
    running_iou = 0.0
    running_f1 = 0.0
    len_data=len(dataset_dl.dataset)

    for xb,yb in dataset_dl:
        xb = xb.to('cuda')
        yb = yb.to('cuda')

        output = model(xb)
        loss_b,meandice,meaniou,meanf1 = util.loss_batch(loss_func,output,yb,opt)
        running_loss += loss_b

        if meandice is not None:
            running_dice += meandice
            running_iou += meaniou
            running_f1 += meanf1
        if sanity_check is True:
            break

    loss = running_loss/float(len_data)
    metric = running_dice/float(len_data)
    avg_iou = running_iou/float(len_data)
    avg_f1 = running_f1/float(len_data)
    # 返回batch的平均损失和度量值
    return loss,metric,avg_iou,avg_f1

#=========================== 训练循环 ===========================#
def train_val(model,params):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    loss_history={
        "train":[],
        "val":[]
    }
    metric_history = {
        "train": [],
        "val": []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr = {}'.format(epoch + 1,num_epochs,current_lr))
        logging.info(f"Epoch {epoch + 1}/{num_epochs},current lr = {current_lr}")

        model.train()
        train_loss,train_metric,train_iou,train_f1 = loss_epoch(model,loss_func,train_dl,sanity_check,opt)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss,val_metric,val_iou,val_f1 = loss_epoch(model,loss_func,val_dl,sanity_check)

        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(),path2weights)
            print("Copied best model weights!")
            logging.info(f"Copied best model weights!")
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            logging.info(f"Loading best model weights!")
            model.load_state_dict(best_model_wts)

        print("train loss: %.6f, dice: %.2f || iou: %.2f || f1: %.2f" %(train_loss,100*train_metric,100*train_iou,100*train_f1))
        print("val loss: %.6f, dice: %.2f || iou: %.2f || f1: %.2f" %(val_loss,100*val_metric,100*val_iou,100*val_f1))
        print("-"*10)
        logging.info(f"train loss: {train_loss:.2f}, dice: {100*train_metric:.2f} || iou: {100*train_iou:.2f} || f1: {100*train_f1:.2f}")
        logging.info(f"val loss:{val_loss:.2f}, dice: {100*val_metric:.2f} || iou: {100*val_iou:.2f} || f1: {100*val_f1:.2f}")
        logging.info(f"-"*10)

    model.load_state_dict(best_model_wts)
    return model,loss_history,metric_history


# 执行训练
model, loss_hist, metric_hist = train_val(model, params_train)

#=========================== 保存最终结果 ===========================#
stopDate = datetime.now().strftime("%Y%m%d_%H_%M_%S")
log_info = {
    "training_params": {
        "dataset": DATASET_NAME,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": OPTIMIZER_PARAMS,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE
    },
    "metrics": {
        "loss_history": loss_hist,
        "metric_history": metric_hist
    },
    "timestamps": {
        "start": startDate,
        "end": stopDate
    }
}

with open(os.path.join(LOG_DIR, LOG_INFO_SAVE_FORMAT.format(startDate)), 'w') as f:
    json.dump(log_info, f, indent=2)

logging.info("Training completed. Total duration: %s", stopDate)
