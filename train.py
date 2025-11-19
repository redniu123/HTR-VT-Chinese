import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # <--- [新增 1] 导入 tqdm

import wandb  # 🔥 [WandB] 导入库

import os
import json
import valid
from utils import utils  # 注意：如果你之前改成了 my_utils，这里要对应修改
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial


# ... (compute_loss 函数保持不变) ...
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
    🔥 [WandB] 辅助函数：抽取验证集图片进行可视化记录
    """
    model.eval()
    images_to_log = []

    # 获取一个 Batch 的数据
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

    # 限制记录数量，防止网页卡顿
    num_images = min(len(labels), max_images)

    # 创建 WandB Table
    table = wandb.Table(columns=["Image", "Ground Truth", "Prediction", "Correct?"])

    for i in range(num_images):
        img = image_tensors[i].cpu().permute(1, 2, 0).numpy() * 255.0  # [C, H, W] -> [H, W, C]
        img = img.squeeze().astype('uint8')  # 灰度图

        gt_text = labels[i]
        pred_text = preds_str[i]
        is_correct = (gt_text == pred_text)

        table.add_data(wandb.Image(img), gt_text, pred_text, is_correct)

    wandb.log({"Predictions/Examples": table}, step=step)
    model.train()


def main():
    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    # 🔥 [WandB] 1. 初始化 (记录超参数)
    if args.use_wandb:
        wandb.init(
            project="HTR-ICDAR2013-Chinese",  # 项目名称，自己在网页上定
            name=args.exp_name,  # 实验名称
            config=vars(args),  # 自动记录所有 argparse 参数
            resume="allow"
        )

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    # 🔥 [WandB] 2. 监控模型梯度和参数分布 (Watch)
    # log="all" 会记录梯度 histograms，非常有用于观察梯度消失/爆炸
    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=1000)

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))
    if args.use_wandb:
        wandb.config.update({"total_params": total_param})  # 补充记录参数量

    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()

    # ... (数据加载代码保持不变) ...
    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                             num_workers=args.num_workers)

    optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=1e-7, betas=(0.9, 0.99),
                        weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0

    # 进度条设置
    iter_per_epoch = len(train_loader)
    pbar = tqdm(range(1, args.total_iter + 1), desc="Training", unit="iter", dynamic_ncols=True)

    for nb_iter in pbar:
        optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda()
        text, length = converter.encode(batch[1])
        batch_size = image.size(0)

        # Forward & Backward
        loss = compute_loss(args, model, image, batch_size, criterion, text, length)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        compute_loss(args, model, image, batch_size, criterion, text, length).backward()
        optimizer.second_step(zero_grad=True)

        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)

        loss_val = loss.item()
        train_loss += loss_val

        # 进度条更新
        if nb_iter % 5 == 0:
            pbar.set_postfix({'Loss': f'{loss_val:.4f}', 'LR': f'{current_lr:.6f}'})

        # 🔥 [WandB] 3. 记录训练指标 (Step-level)
        if args.use_wandb and nb_iter % 10 == 0:
            wandb.log({
                "Train/Loss": loss_val,
                "Train/LR": current_lr,
                "Train/Epoch": nb_iter / iter_per_epoch
            }, step=nb_iter)

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter
            # logger.info(...) # 保持原有日志
            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            pbar.write(f"[{nb_iter}] Starting Validation...")

            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema, criterion, val_loader,
                                                                             converter)

                # 🔥 [WandB] 4. 记录验证指标
                if args.use_wandb:
                    wandb.log({
                        "Val/Loss": val_loss,
                        "Val/CER": val_cer,
                        "Val/WER": val_wer,
                    }, step=nb_iter)

                    # 🔥 [WandB] 5. 可视化预测结果 (图片+GT+预测)
                    # 这是一个非常强大的功能，能让你直观看到模型哪认错了
                    log_predictions_to_wandb(model_ema.ema, val_loader, converter, nb_iter)

                if val_cer < best_cer:
                    pbar.write(f"⭐️ CER improved: {best_cer:.4f} -> {val_cer:.4f}")
                    best_cer = val_cer
                    save_path = os.path.join(args.save_dir, 'best_CER.pth')

                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, save_path)

                    # 🔥 [WandB] 6. 记录最佳指标并上传模型文件 (Artifact)
                    if args.use_wandb:
                        wandb.log({"Val/Best_CER": best_cer}, step=nb_iter)
                        # 创建 Artifact 记录模型文件版本
                        # (注意：如果硬盘空间小或网速慢，可以注释掉下面这两行)
                        # artifact = wandb.Artifact(name=f"model-best-cer-{args.exp_name}", type="model")
                        # artifact.add_file(save_path)
                        # wandb.log_artifact(artifact)

                if val_wer < best_wer:
                    best_wer = val_wer
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))
                    if args.use_wandb:
                        wandb.log({"Val/Best_WER": best_wer}, step=nb_iter)

                pbar.write(f'Val Result @ {nb_iter}: CER={val_cer:.4f} | Loss={val_loss:.3f}')
                model.train()

    # 结束时关闭 wandb
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
