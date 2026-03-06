#!/usr/bin/env python3
# coding: utf-8
"""
Clean script version of your notebook.

Goals:
- identical behavior in script mode vs notebook (as much as possible)
- all knobs (e.g., num_sessions) come from CLI/env, so you can run SLURM arrays
- safe for `accelerate launch` (no heavy work at import time)
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import math
import time
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import webdataset as wds
from accelerate import Accelerator

# --- your repo/local imports ---
import utils

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append("generative_models/")
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

# speed
torch.backends.cuda.matmul.allow_tf32 = True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Model Training Configuration")

    p.add_argument("--model_name", type=str, default="testing")
    p.add_argument("--data_path", type=str,
                   default="/weka/proj-fmri/shared/natural-scenes-dataset")
    p.add_argument("--subj", type=int, default=1, choices=[1,2,3,4,5,6,7,8])
    p.add_argument("--multisubject_ckpt", type=str, default=None)
    p.add_argument("--num_sessions", type=int, default=0)

    p.add_argument("--use_prior", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume_from_ckpt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb_project", type=str, default="stability")

    p.add_argument("--mixup_pct", type=float, default=.33)
    p.add_argument("--low_mem", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--blur_scale", type=float, default=.5)
    p.add_argument("--clip_scale", type=float, default=1.)
    p.add_argument("--prior_scale", type=float, default=30.)
    p.add_argument("--use_image_aug", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--num_epochs", type=int, default=120)
    p.add_argument("--multi_subject", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--hidden_dim", type=int, default=1024)

    p.add_argument("--seq_past", type=int, default=0)
    p.add_argument("--seq_future", type=int, default=0)

    p.add_argument("--lr_scheduler_type", type=str, default="cycle", choices=["cycle","linear"])
    p.add_argument("--ckpt_saving", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ckpt_interval", type=int, default=5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--prior_lr", type=float, default=None)

    # output root so you can point runs wherever you want
    p.add_argument("--out_root", type=str,
                   default="/teamspace/gcs_folders/share/real_time_mindeye_data/3t_data/data/model")

    return p.parse_args(argv)


def set_all_seeds(seed: int, rank: int = 0, deterministic: bool = False):
    """Seed python/numpy/torch. Optionally add rank offset."""
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_batch_sizes(args, accelerator: Accelerator):
    """
    Choose global_batch_size and per-process batch_size.

    - In notebook you used env GLOBAL_BATCH_SIZE.
    - For script, default to env if set, else args.batch_size.
    """
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        num_devices = 1

    # prefer env var for sweeps / SLURM scripts
    if os.getenv("GLOBAL_BATCH_SIZE") is not None:
        global_batch_size = int(os.getenv("GLOBAL_BATCH_SIZE"))
    else:
        # interpret args.batch_size as global if user passes it (common)
        global_batch_size = int(args.batch_size)

    # per-process (each process sees batch_size)
    per_proc_bs = max(1, global_batch_size // max(1, num_devices))

    return global_batch_size, per_proc_bs, num_devices


def build_outdir(args):
    outdir = os.path.abspath(os.path.join(args.out_root, args.model_name))
    if args.ckpt_saving:
        os.makedirs(outdir, exist_ok=True)
    return outdir


def main(argv=None):
    args = parse_args(argv)

    # Accelerator should be created early, but after args are known
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == "NO"

    # Keep prints sane: use accelerator.print instead of overwriting print()
    aprint = accelerator.print

    # Seeds: If you want each rank to be different but reproducible, use rank offset
    set_all_seeds(args.seed, rank=rank, deterministic=False)

    # Batch sizing
    global_batch_size, batch_size, num_devices = get_batch_sizes(args, accelerator)

    # Basic run info
    aprint(f"PID={os.getpid()} device={device}")
    aprint(f"distributed={distributed} num_devices={num_devices} rank={rank}/{world_size}")
    aprint(f"GLOBAL_BATCH_SIZE={global_batch_size} per_process_batch_size={batch_size}")
    aprint("args:\n", args)

    # Output
    outdir = build_outdir(args)
    if accelerator.is_main_process:
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # Optional kornia aug imports
    if args.use_image_aug or args.blurry_recon:
        import kornia
        import kornia.augmentation as K
        from kornia.augmentation.container import AugmentationSequential

    if args.use_image_aug:
        img_augment = AugmentationSequential(
            # same as your notebook
            kornia.augmentation.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3
            ),
            same_on_batch=False,
            data_keys=["input"],
        )
        blur_augment = K.RandomGaussianBlur(kernel_size=(21, 21), sigma=(51.0, 51.0), p=1.0)
    else:
        img_augment = None
        blur_augment = None

    # Subject list logic
    if args.multi_subject:
        subj_list = np.arange(1, 9)
        subj_list = subj_list[subj_list != args.subj]
    else:
        subj_list = [args.subj]
    aprint(f"subj_list={subj_list} num_sessions={args.num_sessions}")

    # ---- Data size logic ----
    if args.multi_subject:
        nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
        num_samples_per_epoch = (750 * 40) // max(1, num_devices)
    else:
        num_samples_per_epoch = (750 * args.num_sessions) // max(1, num_devices)

    # divide batch size across subjects (your original behavior)
    batch_size = max(1, batch_size // len(subj_list))

    # IMPORTANT: avoid 0 iterations
    denom = batch_size * len(subj_list)
    num_iterations_per_epoch = max(1, num_samples_per_epoch // max(1, denom))

    aprint(f"batch_size(after subj split)={batch_size}")
    aprint(f"num_samples_per_epoch={num_samples_per_epoch} num_iterations_per_epoch={num_iterations_per_epoch}")

    # ---- WebDataset loaders + betas ----
    def my_split_by_node(urls):  # keep same
        return urls

    train_data = {}
    train_dl = {}
    num_voxels = {}
    voxels = {}
    num_voxels_list = []

    for s in subj_list:
        aprint(f"Training with num_sessions={args.num_sessions} subj0{s}")

        if args.multi_subject:
            train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
        else:
            train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{args.num_sessions-1}" + "}.tar"
        aprint("train_url:", train_url)

        train_data[f"subj0{s}"] = (
            wds.WebDataset(train_url, resampled=True, nodesplitter=my_split_by_node)
            .shuffle(750, initial=1500, rng=random.Random(42))
            .decode("torch")
            .rename(
                behav="behav.npy",
                past_behav="past_behav.npy",
                future_behav="future_behav.npy",
                olds_behav="olds_behav.npy",
            )
            .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
        )
        train_dl[f"subj0{s}"] = torch.utils.data.DataLoader(
            train_data[f"subj0{s}"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0,  # keep 0 if you want closer behavior to notebook
        )

        with h5py.File(f"{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5", "r") as f:
            betas = f["betas"][:]
        betas = torch.tensor(betas).to("cpu").to(torch.float16)  # keep same type
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f"subj0{s}"] = betas[0].shape[-1]
        voxels[f"subj0{s}"] = betas
        aprint(f"num_voxels subj0{s} = {num_voxels[f'subj0{s}']}")

    aprint("Loaded all subj train dls and betas!")

    # Validate only on one subject (same logic)
    subj_for_test = args.subj
    if args.multi_subject:
        subj_for_test = int(subj_list[0])

    if not args.new_test:
        if subj_for_test in (3, 6):
            num_test = 2113
        elif subj_for_test in (4, 8):
            num_test = 1985
        else:
            num_test = 2770
        test_url = f"{args.data_path}/wds/subj0{subj_for_test}/test/0.tar"
    else:
        if subj_for_test in (3, 6):
            num_test = 2371
        elif subj_for_test in (4, 8):
            num_test = 2188
        else:
            num_test = 3000
        test_url = f"{args.data_path}/wds/subj0{subj_for_test}/new_test/0.tar"

    aprint("test_url:", test_url)
    test_data = (
        wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
        .shuffle(750, initial=1500, rng=random.Random(42))
        .decode("torch")
        .rename(
            behav="behav.npy",
            past_behav="past_behav.npy",
            future_behav="future_behav.npy",
            olds_behav="olds_behav.npy",
        )
        .to_tuple("behav", "past_behav", "future_behav", "olds_behav")
    )
    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True, num_workers=0
    )
    aprint(f"Loaded test dl for subj{subj_for_test}")

    # Load 73k images (cpu)
    with h5py.File(f"{args.data_path}/coco_images_224_float16.hdf5", "r") as f:
        images = f["images"]  # h5py dataset handle
        aprint("Loaded coco_images_224_float16.hdf5 images:", images.shape)

        # ---- CLIP image embedder ----
        clip_img_embedder = FrozenOpenCLIPImageEmbedder(
            arch="ViT-bigG-14",
            version="laion2b_s39b_b160k",
            output_tokens=True,
            only_tokens=True,
        ).to(device)

        clip_seq_dim = 256
        clip_emb_dim = 1664

        # ---- Model ----
        model = utils.prepare_model_and_training(
            num_voxels_list=num_voxels_list,
            n_blocks=args.n_blocks,
            hidden_dim=args.hidden_dim,
            clip_emb_dim=clip_emb_dim,
            clip_seq_dim=clip_seq_dim,
            use_prior=args.use_prior,
            clip_scale=args.clip_scale,
        )

        # ---- Optimizer / scheduler ----
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        opt_grouped_parameters = [
            {"params": [p for _, p in model.ridge.named_parameters()], "weight_decay": 1e-2},
            {"params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 1e-2},
            {"params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if args.use_prior:
            effective_prior_lr = args.prior_lr if args.prior_lr is not None else args.max_lr
            aprint(f"Setting diffusion_prior lr={effective_prior_lr}")
            if args.prior_lr is not None:
                assert args.lr_scheduler_type == "cycle"
            opt_grouped_parameters.extend([
                {"params": [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)],
                 "weight_decay": 1e-2, "lr": effective_prior_lr},
                {"params": [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0, "lr": effective_prior_lr},
            ])

        optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

        if args.lr_scheduler_type == "linear":
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                total_iters=int(np.floor(args.num_epochs * num_iterations_per_epoch)),
                last_epoch=-1,
            )
        else:
            total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
            max_lrs = [args.max_lr] * 3
            if args.use_prior:
                max_lrs.extend([effective_prior_lr] * 2)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=max(1, total_steps),
                final_div_factor=1000,
                last_epoch=-1,
                pct_start= max(0.01, min(0.3, 2 / float(args.num_epochs))),
            )

        # ---- ckpt helpers ----
        losses, test_losses, lrs = [], [], []

        def save_ckpt(tag, epoch):
            if not args.ckpt_saving:
                return
            ckpt_path = os.path.join(outdir, f"{tag}.pth")
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_losses": losses,
                        "test_losses": test_losses,
                        "lrs": lrs,
                    },
                    ckpt_path,
                )
            aprint(f"saved ckpt: {ckpt_path}")

        def load_ckpt(outdir_to_load, tag="last", strict=True, multisubj_loading=False,
                      load_lr=True, load_optimizer=True, load_epoch=False):
            aprint(f"loading ckpt: {outdir_to_load}/{tag}.pth")
            checkpoint = torch.load(os.path.join(outdir_to_load, f"{tag}.pth"), map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            if multisubj_loading:
                state_dict.pop("ridge.linears.0.weight", None)
            model.load_state_dict(state_dict, strict=strict)
            if load_optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if load_lr:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = 0
            if load_epoch:
                start_epoch = checkpoint["epoch"]
            del checkpoint
            return start_epoch

        # load multisubject ckpt if requested
        start_epoch = 0
        if args.multisubject_ckpt is not None and (not args.resume_from_ckpt):
            start_epoch = load_ckpt(
                args.multisubject_ckpt,
                tag="last",
                strict=False,
                multisubj_loading=True,
                load_lr=False,
                load_optimizer=False,
                load_epoch=False,
            )

        # ---- Prepare with accelerator ----
        train_dls = [train_dl[f"subj0{s}"] for s in subj_list]
        model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)

        # ---- Train loop (your code, moved verbatim-ish) ----
        aprint(f"{args.model_name} starting epoch {start_epoch}/{args.num_epochs}")
        progress_bar = tqdm(range(start_epoch, args.num_epochs), ncols=120, disable=(rank != 0))

        mse = nn.MSELoss()
        l1 = nn.L1Loss()
        soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))

        test_image, test_voxel = None, None

        for epoch in progress_bar:
            model.train()

            # (keep your metrics init exactly)
            fwd_percent_correct = 0.0
            bwd_percent_correct = 0.0
            test_fwd_percent_correct = 0.0
            test_bwd_percent_correct = 0.0

            recon_cossim = 0.0
            test_recon_cossim = 0.0
            recon_mse = 0.0
            test_recon_mse = 0.0

            loss_clip_total = 0.0
            loss_blurry_total = 0.0
            loss_blurry_cont_total = 0.0
            test_loss_clip_total = 0.0

            loss_prior_total = 0.0
            test_loss_prior_total = 0.0

            blurry_pixcorr = 0.0
            test_blurry_pixcorr = 0.0

            # ---- PRELOAD BATCHES (your logic) ----
            voxel_iters = {}
            image_iters = torch.zeros(
                num_iterations_per_epoch, batch_size * len(subj_list), 3, 224, 224
            ).float()

            perm_iters, betas_iters, select_iters = {}, {}, {}

            for si, train_dl_one in enumerate(train_dls):
                iter_i = -1
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    for behav0, past_behav0, future_behav0, old_behav0 in train_dl_one:
                        image_idx = behav0[:, 0, 0].cpu().long().numpy()
                        image0, image_sorted_idx = np.unique(image_idx, return_index=True)
                        if len(image0) != len(image_idx):
                            continue
                        iter_i += 1

                        # load images from hdf5 (sorted unique indexing)
                        image0_t = torch.tensor(images[image0], dtype=torch.float16)
                        image_iters[iter_i, si * batch_size: si * batch_size + batch_size] = image0_t

                        # voxels
                        voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
                        voxel_sorted_idx = voxel_idx[image_sorted_idx]
                        voxel0 = voxels[f"subj0{subj_list[si]}"][voxel_sorted_idx]
                        voxel0 = torch.tensor(voxel0).unsqueeze(1)

                        if epoch < int(args.mixup_pct * args.num_epochs):
                            voxel0, perm, betas, select = utils.mixco(voxel0)
                            key = f"subj0{subj_list[si]}_iter{iter_i}"
                            perm_iters[key] = perm
                            betas_iters[key] = betas
                            select_iters[key] = select

                        voxel_iters[f"subj0{subj_list[si]}_iter{iter_i}"] = voxel0

                        if iter_i >= num_iterations_per_epoch - 1:
                            break

            # ---- TRAIN STEPS ----
            for train_i in range(num_iterations_per_epoch):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    optimizer.zero_grad()
                    loss = 0.0

                    voxel_list = [
                        voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device)
                        for s in subj_list
                    ]

                    image = image_iters[train_i].detach().to(device)

                    if img_augment is not None:
                        image = img_augment(image)

                    clip_target = clip_img_embedder(image)
                    assert not torch.any(torch.isnan(clip_target))

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        perm = torch.cat(
                            [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list],
                            dim=0,
                        )
                        betas = torch.cat(
                            [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list],
                            dim=0,
                        )
                        select = torch.cat(
                            [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list],
                            dim=0,
                        )

                    voxel_ridge_list = [model.ridge(voxel_list[si], si) for si, _ in enumerate(subj_list)]
                    voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                    backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

                    if args.clip_scale > 0:
                        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                    if args.use_prior:
                        loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                        loss_prior_total += float(loss_prior.item())
                        loss += loss_prior * args.prior_scale
                        recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                        recon_mse += mse(prior_out, clip_target).item()

                    if args.clip_scale > 0:
                        if epoch < int(args.mixup_pct * args.num_epochs):
                            loss_clip = utils.mixco_nce(
                                clip_voxels_norm, clip_target_norm, temp=0.006,
                                perm=perm, betas=betas, select=select
                            )
                        else:
                            epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                            loss_clip = utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=epoch_temp)

                        loss_clip_total += float(loss_clip.item())
                        loss += loss_clip * args.clip_scale

                        labels = torch.arange(len(clip_voxels_norm), device=clip_voxels_norm.device)
                        fwd_percent_correct += utils.topk(
                            utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm),
                            labels, k=1
                        ).item()
                        bwd_percent_correct += utils.topk(
                            utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm),
                            labels, k=1
                        ).item()

                    # NOTE: your blurry_recon branch depends on autoenc/cnx/mean/std/blur_augs
                    # Keep it verbatim here once you paste those definitions above.
                    # if args.blurry_recon:
                    #     ... (paste your existing block) ...

                    utils.check_loss(loss)
                    accelerator.backward(loss)
                    optimizer.step()

                    losses.append(float(loss.item()))
                    lrs.append(float(optimizer.param_groups[0]["lr"]))
                    if args.lr_scheduler_type is not None:
                        lr_scheduler.step()

            # ---- EVAL (rank 0 only, same structure) ----
            model.eval()
            if rank == 0:
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                    for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
                        assert len(behav) == num_test

                        # build cached test_image/test_voxel (your logic)
                        if test_image is None:
                            voxel = voxels[f"subj0{subj_for_test}"][behav[:, 0, 5].cpu().long()].unsqueeze(1)
                            image_ids = behav[:, 0, 0].cpu().long()

                            unique_image = torch.unique(image_ids)
                            for im in unique_image:
                                locs = torch.where(im == image_ids)[0]
                                if len(locs) == 1:
                                    locs = locs.repeat(3)
                                elif len(locs) == 2:
                                    locs = locs.repeat(2)[:3]
                                assert len(locs) == 3

                                im_t = torch.tensor(images[int(im)][None])
                                vox_t = voxel[locs][None]
                                test_image = im_t if test_image is None else torch.vstack((test_image, im_t))
                                test_voxel = vox_t if test_voxel is None else torch.vstack((test_voxel, vox_t))

                        loss_eval = 0.0
                        test_indices = torch.arange(len(test_voxel))[:300]
                        voxel = test_voxel[test_indices].to(device)
                        image = test_image[test_indices].to(device)
                        assert len(image) == 300

                        clip_target = clip_img_embedder(image.float())

                        for rep in range(3):
                            voxel_ridge = model.ridge(voxel[:, rep], 0)
                            backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)
                            if rep == 0:
                                clip_voxels = clip_voxels0
                                backbone = backbone0
                            else:
                                clip_voxels += clip_voxels0
                                backbone += backbone0
                        clip_voxels /= 3
                        backbone /= 3

                        if args.clip_scale > 0:
                            clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                        random_samps = np.random.choice(np.arange(len(image)), size=max(1, len(image)//5), replace=False)

                        if args.use_prior:
                            loss_prior, _ = model.diffusion_prior(
                                text_embed=backbone[random_samps],
                                image_embed=clip_target[random_samps],
                            )
                            test_loss_prior_total += float(loss_prior.item())
                            loss_eval += float((loss_prior * args.prior_scale).item())

                        if args.clip_scale > 0:
                            loss_clip = utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=0.006)
                            test_loss_clip_total += float(loss_clip.item())
                            loss_eval += float((loss_clip * args.clip_scale).item())

                        # NOTE: paste your blurry_recon eval branch here if needed

                        test_losses.append(loss_eval)

                    # logs + ckpt
                    logs = {
                        "train/loss": float(np.mean(losses[-(train_i+1):])),
                        "test/loss": float(np.mean(test_losses[-(test_i+1):])),
                        "train/lr": float(lrs[-1]),
                        "train/fwd_pct_correct": float(fwd_percent_correct / (train_i + 1)),
                        "train/bwd_pct_correct": float(bwd_percent_correct / (train_i + 1)),
                    }
                    progress_bar.set_postfix(**logs)

            if args.ckpt_saving and (epoch % args.ckpt_interval == 0):
                save_ckpt("last", epoch)

            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

        aprint("=== Finished training ===")
        save_ckpt("last", args.num_epochs - 1)

    # rank 0 plots (optional)
    if accelerator.is_main_process:
        plt.figure()
        plt.plot(losses)
        plt.savefig(os.path.join(outdir, "losses.png"), dpi=150)
        plt.close()

        plt.figure()
        plt.plot(test_losses)
        plt.savefig(os.path.join(outdir, "test_losses.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
