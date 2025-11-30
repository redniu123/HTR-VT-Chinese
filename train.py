import os
import json
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from functools import partial
import warnings
import torch.multiprocessing
import valid
import wandb
from torch.cuda.amp import autocast, GradScaler

try:
    from utils import my_utils as utils
except ImportError:
    from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT

# 1. 解决 Bus Error
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except:
    pass
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
warnings.filterwarnings("ignore")

def compute_loss(args, model, image, batch_size, criterion, text, length):
    preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    preds = preds.float()
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
    preds = preds.permute(1, 0, 2).log_softmax(2)
    torch.backends.cudnn.enabled = False
    loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True
    return loss

def log_predictions_to_wandb(model, val_loader, converter, step, max_images=8):
    """
    [WandB] 可视化：这里会生成一个 Table，包含图片、GT、预测值
    """
    if not wandb.run: return
    model.eval()
    try:
        image_tensors, labels = next(iter(val_loader))
    except StopIteration:
        return

    image_tensors = image_tensors.cuda()
    batch_size = image_tensors.size(0)
    
    with torch.no_grad():
        preds = model(image_tensors)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)

    table = wandb.Table(columns=["Image", "Ground Truth", "Prediction", "Match"])
    
    # 记录前 N 张
    num_log = min(len(labels), max_images)
    for i in range(num_log):
        img = image_tensors[i].cpu().permute(1, 2, 0).numpy() * 255.0
        img = img.squeeze().astype('uint8')
        gt_text = labels[i]
        pred_text = preds_str[i]
        is_match = (gt_text == pred_text)
        table.add_data(wandb.Image(img), gt_text, pred_text, is_match)

    # 提交 Table
    wandb.log({"Val/Predictions": table}, step=step)
    model.train()

def main():
    args = option.get_args_parser()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    if args.use_wandb:
        wandb.init(project="HTR-ICDAR2013-Chinese", name=args.exp_name, config=vars(args))

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
    
    # 你的 Local Attention 应该在 HTR_VT 内部已经被调用了
    
    # --- [关键修改] 单卡训练模式 (避免多卡 NCCL 错误) ---
    # 如果你想试多卡，可以取消注释，但鉴于之前的报错，单卡最稳
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=True)

# --- [新增] 断点续训 (Resume) 功能 ---
    start_iter = 1
    checkpoint_path = os.path.join(args.save_dir, 'best_CER.pth')
    
    if os.path.exists(checkpoint_path):
        logger.info(f"🔄 Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path)
        
        # 1. 加载模型权重
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        
        # 2. 加载优化器状态 (这很重要，保证学习率衔接)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        # 3. 加载混合精度状态
        if 'scaler' in checkpoint and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
            
        # 4. 恢复迭代次数 (如果有记录的话，没有就估算)
        if 'iter' in checkpoint:
            start_iter = checkpoint['iter'] + 1
            logger.info(f"Resuming from iteration {start_iter}")
        else:
            logger.info("Warning: No iteration info in checkpoint, starting from iter 1 but with pretrained weights.")
    else:
        logger.info("No checkpoint found, starting from scratch.")

    # ... (之后的代码) ...


    # 数据加载
    logger.info('Loading train dataset...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    
    # 训练集：开启多进程 (因为有 file_system 策略)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_bs,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers, # 命令行传参
        collate_fn=partial(dataset.SameTrCollate, args=args)
    )
    train_iter = dataset.cycle_data(train_loader)

    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    best_cer = 1e+6
    iter_per_epoch = len(train_loader)
    
# 修改循环范围：从 start_iter 开始，而不是从 1 开始
    logger.info(f"Start Training from {start_iter} to {args.total_iter}...")
    pbar = tqdm(range(start_iter, args.total_iter + 1), desc="Training", unit="iter", dynamic_ncols=True)

    # logger.info(f"Start Training for {args.total_iter} iterations...")
    # pbar = tqdm(range(1, args.total_iter + 1), desc="Training", unit="iter", dynamic_ncols=True)

    for nb_iter in pbar:
        model.train()
        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)
        optimizer.zero_grad()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = dataset.cycle_data(train_loader)
            batch = next(train_iter)

        image = batch[0].cuda()
        text, length = converter.encode(batch[1])
        batch_size = image.size(0)
        
        with autocast():
            loss = compute_loss(args, model, image, batch_size, criterion, text, length)
        
        scaler.scale(loss).backward()
        
        # --- 🔥 [关键修复] 梯度裁剪 (Gradient Clipping) 🔥 ---
        # 防止 Loss 突然爆炸，这是解决那张图里 Spike 的关键
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        # ----------------------------------------------------
        
        scaler.step(optimizer)
        scaler.update()
        
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        
        if nb_iter % 5 == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{current_lr:.6f}'})

        if args.use_wandb and nb_iter % 50 == 0:
            wandb.log({"Train/Loss": loss.item(), "Train/LR": current_lr}, step=nb_iter)

        if nb_iter % args.eval_iter == 0:
            pbar.write(f"\n[{nb_iter}] Validating...")
            val_loss, val_cer, val_wer, _, _ = valid.validation(model_ema.ema, criterion, val_loader, converter)
            
            if args.use_wandb:
                wandb.log({"Val/Loss": val_loss, "Val/CER": val_cer}, step=nb_iter)
                log_predictions_to_wandb(model_ema.ema, val_loader, converter, nb_iter)

            if val_cer < best_cer:
                best_cer = val_cer
                pbar.write(f"⭐️ Best CER: {best_cer:.4f}")
                # 保存模型... (代码略，同前)
                torch.save({'model': model.state_dict(), 'best_cer': best_cer}, os.path.join(args.save_dir, 'best_CER.pth'))
                if args.use_wandb: wandb.log({"Val/Best_CER": best_cer}, step=nb_iter)

    if args.use_wandb: wandb.finish()

if __name__ == '__main__':
    main()