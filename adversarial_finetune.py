import argparse
import datetime
import json
import os
import time
from pathlib import Path
import warnings
import faulthandler
# =========================
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from huggingface_hub import hf_hub_download, login
# =========================
import models_vit as models
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch_adversarial, evaluate
import torch.nn as nn
# =========================
faulthandler.enable()
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning / linear probing for image classification", add_help=False
    )

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--label_col", type=str, required=True)
    parser.add_argument("--private_label_col", type=str, required=True)
    parser.add_argument("--filename_col", type=str, default="filename")

    parser.add_argument("--data_path", default="./data/", type=str)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--val_ratio",   default=0.0, type=float)
    parser.add_argument("--test_ratio",  default=0.2, type=float)
    # ---- Core training
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    # ---- Model parameters
    parser.add_argument("--model", default="RETFound_mae", type=str, metavar="MODEL")
    parser.add_argument("--model_arch", default="dinov3_vits16", type=str, metavar="MODEL_ARCH")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--drop_path", type=float, default=0.2, metavar="PCT")
    parser.add_argument("--global_pool", action="store_true"); parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool")
    # ---- Optimizer parameters
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=5e-3, metavar="LR")
    parser.add_argument("--layer_decay", type=float, default=0.65)
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N")
    # ---- Augmentation
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME")
    parser.add_argument("--smoothing", type=float, default=0.1)
    # ---- Random erase
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT")
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--resplit", action="store_true", default=False)
    # ---- Mixup/Cutmix
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--cutmix_minmax", type=float, nargs="+", default=None)
    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_switch_prob", type=float, default=0.5)
    parser.add_argument("--mixup_mode", type=str, default="batch")
    # ---- Finetuning & adaptation
    parser.add_argument("--finetune", default="", type=str)
    parser.add_argument("--task", default="", type=str)
    parser.add_argument("--nb_private_classes", default=2, type=int)
    parser.add_argument("--lambda_adv", default=1.0, type=float)
    parser.add_argument("--adaptation", default="adversarial",
                        choices=["finetune", "lp", "adversarial"])
    # ---- Classifier weights
    parser.add_argument("--c_weights", default="", type=str, help="Path to classifier weights")
    parser.add_argument("--ac_weights", default="", type=str, help="Path to adversary classifier weights")
    # ---- Dataset & paths
    parser.add_argument("--nb_classes", default=8, type=int)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--log_dir", default="./output_logs")
    # ---- Training data efficiency
    parser.add_argument("--dataratio", type=str, default="1.0")
    parser.add_argument("--stratified", action="store_true")
    # ---- Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true"); parser.set_defaults(pin_mem=True)
    # ---- Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    # ---- Misc
    parser.add_argument("--savemodel", action="store_true", default=True)
    parser.add_argument("--norm", default="IMAGENET", type=str)
    parser.add_argument("--enhance", action="store_true", default=False)
    parser.add_argument("--datasets_seed", default=2026, type=int)
    parser.add_argument("--save_prefix", type=str, default="",
                        help="Prefix prepended to saved checkpoint filenames")

    parser.add_argument("--hidden1", type=int, help="hidden layer 1 neurons")
    parser.add_argument("--hidden2", type=int, help="hidden layer 2 neurons")
    return parser


# =========================
# Main
# =========================
def main(args, criterion):
    # ---- Optionally load args from resume (when training)
    if args.resume and not args.eval:
        resume_path = args.resume
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"Load checkpoint (args) from: {args.resume}")
        args = checkpoint["args"]
        args.resume = resume_path

    # ---- Distributed setup
    misc.init_distributed_mode(args)
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)

    # ---- Reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- Build model
    if args.model == "RETFound_mae":
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )

    # ---- Load pre-trained weights (if requested and not eval-only)
    if args.finetune and not args.eval:
        print(f"Preparing to load pre-trained weights: {args.finetune}")
        if args.model in ["Dinov3", "Dinov2"]:
            checkpoint_path = args.finetune
        elif args.model in ["RETFound_dinov2", "RETFound_mae"]:
            print(f"Downloading pre-trained weights from Hugging Face Hub: {args.finetune}")
            checkpoint_path = hf_hub_download(
                repo_id=f"YukunZhou/{args.finetune}",
                filename=f"{args.finetune}.pth",
            )
        else:
            raise ValueError(
                f"Unsupported model '{args.model}'. "
                f"Expected one of: Dinov3, Dinov2, RETFound_dinov2, RETFound_mae"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loaded pre-trained checkpoint from: {checkpoint_path}")
        if args.model in ["Dinov3", "Dinov2"]:
            checkpoint_model = checkpoint
        elif args.model == "RETFound_dinov2":
            checkpoint_model = checkpoint["teacher"]
        else:
            checkpoint_model = checkpoint["model"]
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(model, checkpoint_model)
        _ = model.load_state_dict(checkpoint_model, strict=False)
        if hasattr(model, "head") and hasattr(model.head, "weight"):
            trunc_normal_(model.head.weight, std=2e-5)

    # ---- Datasets & samplers
    dataset_train = build_dataset(is_train="train", args=args)
    dataset_val   = build_dataset(is_train="val",   args=args)
    dataset_test  = build_dataset(is_train="test",  args=args)

    num_tasks   = misc.get_world_size()
    global_rank = misc.get_rank()

    if not args.eval:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print(f"Sampler_train = {sampler_train}")
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print("Warning: dist eval with dataset not divisible by #procs.")
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.dist_eval:
        if len(dataset_test) % num_tasks != 0:
            print("Warning: dist eval test set not divisible by #procs.")
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # ---- Logging
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    # ---- DataLoaders
    if not args.eval:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,
        )
        print(f"len of train_set: {len(data_loader_train) * args.batch_size}")

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # ---- Mixup/CutMix
    mixup_fn = None
    mixup_active = (args.mixup > 0) or (args.cutmix > 0.) or (args.cutmix_minmax is not None)
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes
        )

    # ---- Eval-only: resume weights
    if args.resume and args.eval:
        checkpoint = torch.load(args.resume, map_location="cpu")
        print(f"Load checkpoint for eval from: {args.resume}")
        model.load_state_dict(checkpoint["model"])

    model.to(device)
    model_without_ddp = model

    # ---- All encoder weights trainable
    for name, param in model.named_parameters():
        param.requires_grad = True
    print("[Adaptation] Adversarial finetuning: all encoder weights trainable.")

    # ---- Count trainable params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params (M): {n_parameters / 1.e6:.2f}")

    # ---- LR scaling by effective batch size
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print(f"base lr: {args.lr * 256 / eff_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"accumulate grad iterations: {args.accum_iter}")
    print(f"effective batch size: {eff_batch_size}")

    # ---- DDP (if available)
    if args.distributed and torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # ---- Build classifier and adversary classifier
    classifier = nn.Linear(1024, args.nb_classes).to(device)
    adversary_classifier = nn.Sequential(
        nn.Linear(1024, args.hidden1),
        nn.ReLU(),
        nn.Linear(args.hidden1, args.hidden2),
        nn.ReLU(),
        nn.Linear(args.hidden2, args.nb_private_classes)
    ).to(device)

    # ---- Load pretrained classifier and adversary weights
    if args.c_weights:
        classifier.load_state_dict(torch.load(args.c_weights, map_location="cpu"))
        print(f"Loaded classifier weights from: {args.c_weights}")
    if args.ac_weights:
        adversary_classifier.load_state_dict(torch.load(args.ac_weights, map_location="cpu"))
        print(f"Loaded adversary classifier weights from: {args.ac_weights}")

    # ---- Optimizers (one per component)
    no_weight_decay = (model_without_ddp.no_weight_decay()
                       if hasattr(model_without_ddp, "no_weight_decay") else [])
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        weight_decay=args.weight_decay,
        no_weight_decay_list=no_weight_decay,
        layer_decay=args.layer_decay,
    )
    for g in param_groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]

    optimizer_enc = torch.optim.AdamW(param_groups, lr=args.lr)
    optimizer_cls = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_adv = torch.optim.AdamW(adversary_classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion_adv = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    print(f"criterion = {criterion}")

    # ---- Load previous encoder state if resuming
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer_enc, loss_scaler=loss_scaler)

    # =========================
    # Eval-only Short Circuit
    # =========================
    if args.eval:
        if "checkpoint" in locals() and isinstance(checkpoint, dict) and ("epoch" in checkpoint):
            print(f"Test with the best model at epoch = {checkpoint['epoch']}")
        test_stats, auc_roc = evaluate(
            data_loader_test, model, device, args, epoch=0, mode="test",
            num_class=args.nb_classes, log_writer=log_writer
        )
        return

    # =========================
    # Train Loop
    # =========================
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_score = 0.0
    best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_adversarial(
            model=model,
            classifier=classifier,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer_enc=optimizer_enc,
            optimizer_cls=optimizer_cls,
            optimizer_adv=optimizer_adv,
            adversary_classifier=adversary_classifier,
            criterion_adv=criterion_adv,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            mixup_fn=mixup_fn,
            log_writer=log_writer,
            args=args,
        )

        val_stats, val_score = evaluate(
            data_loader_train, model, device, args, epoch, mode="val",
            num_class=args.nb_classes, log_writer=log_writer
        )

        if max_score < val_score:
            max_score = val_score
            best_epoch = epoch
            if args.output_dir and args.savemodel:
                prefix = f"{args.save_prefix}_" if args.save_prefix else ""
                torch.save(
                    {"model": model_without_ddp.state_dict(), "epoch": epoch, "args": args},
                    os.path.join("checkpoints", "encoder", f"{prefix}_encoder-best.pth"),
                )
                torch.save(classifier.state_dict(),
                           os.path.join("checkpoints", "linear", f"{prefix}_classifier-best.pth"))
                torch.save(adversary_classifier.state_dict(),
                           os.path.join("checkpoints", "mlp", f"{prefix}_adversary-best.pth"))

        print(f"Best epoch = {best_epoch}, Best score = {max_score:.4f}")

        if log_writer is not None:
            log_writer.add_scalar("loss/val", val_stats["loss"], epoch)
            log_writer.flush()

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                     "epoch": epoch,
                     "n_parameters": n_parameters}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, args.task, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # =========================
    # Final Test (Best Ckpt)
    # =========================
    prefix = f"{args.save_prefix}_" if args.save_prefix else ""
    ckpt_path = os.path.join("checkpoints", "encoder", f"{prefix}_encoder-best.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    print(f"Test with the best model, epoch = {checkpoint.get('epoch', -1)}:")
    _test_stats, _auc_roc = evaluate(
        data_loader_test, model, device, args, -1, mode="test",
        num_class=args.nb_classes, log_writer=None
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    criterion = torch.nn.CrossEntropyLoss()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, criterion)
