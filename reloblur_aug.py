import argparse
import os
import logging
import random
import time
import datetime
from itertools import islice

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

import pyiqa

from models.diffusion_model import UNet
from models.diffusion_network import DDIM, DDPM
from utils.set_condition import select_condition_strategy
from utils.flow_viz import flow_to_image
from utils.utils import (
    same_seed,
    count_parameters,
    tensor2cv,
    AverageMeter,
)


class PairMaskConditionLoader(Dataset):
    """
    适配如下目录结构：

    data_path/
      train/
        s01/
          00/
            00_blur.png
            00_sharp.png
            01_blur.png
            01_sharp.png
      test/
        scene7/
          00/
            00_blur.png
            00_sharp.png

    mask_path/
      train/
        s01/
          00/
            00.png
            01.png
      test/
        scene7/
          00/
            00.png

    返回字段：
      blur, sharp, flow, mask, rel_dir, stem

    这里的 flow 不再是磁盘读取的真实 flow，
    而是由 mask 伪造的“基础 blur condition”，
    后续仍然交给 select_condition_strategy 继续处理。
    """

    def __init__(
        self,
        data_path,
        mask_path,
        mode="train",
        crop_size=None,
        invert_mask=False,
        mask_blur_ksize=21,
        base_mag_min=0.30,
        base_mag_max=0.60,
        base_angle_mode="deterministic_by_index",
    ):
        self.data_path = data_path
        self.mask_path = mask_path
        self.mode = mode
        self.crop_size = crop_size
        self.invert_mask = invert_mask
        self.mask_blur_ksize = mask_blur_ksize
        self.base_mag_min = base_mag_min
        self.base_mag_max = base_mag_max
        self.base_angle_mode = base_angle_mode

        self.samples = []
        self.blur_list = []

        mode_root = os.path.join(data_path, mode)
        if not os.path.exists(mode_root):
            raise FileNotFoundError(f"{mode_root} does not exist")

        for root, dirs, files in os.walk(mode_root):
            sharp_files = sorted([f for f in files if f.endswith("_sharp.png")])
            if len(sharp_files) == 0:
                continue

            rel_dir = os.path.relpath(root, mode_root)

            for sharp_name in sharp_files:
                stem = sharp_name.replace("_sharp.png", "")
                blur_name = f"{stem}_blur.png"
                mask_name = f"{stem}.png"

                blur_path = os.path.join(root, blur_name)
                sharp_path = os.path.join(root, sharp_name)
                mask_file = os.path.join(mask_path, mode, rel_dir, mask_name)

                if not os.path.exists(blur_path):
                    continue
                if not os.path.exists(sharp_path):
                    continue
                if not os.path.exists(mask_file):
                    continue

                self.samples.append({
                    "blur_path": blur_path,
                    "sharp_path": sharp_path,
                    "mask_path": mask_file,
                    "rel_dir": rel_dir,
                    "stem": stem
                })
                self.blur_list.append(blur_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {mode_root}")

    def __len__(self):
        return len(self.samples)

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1) - 0.5  # [3,H,W], range [-0.5, 0.5]
        return img

    def _load_mask(self, path, size_hw=None):
        mask = Image.open(path).convert("L")
        if size_hw is not None:
            h, w = size_hw
            mask = mask.resize((w, h), Image.NEAREST)

        mask = np.array(mask).astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        if self.invert_mask:
            mask = 1.0 - mask

        if self.mask_blur_ksize is not None and self.mask_blur_ksize > 1:
            k = self.mask_blur_ksize
            if k % 2 == 0:
                k += 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            mask = np.clip(mask, 0.0, 1.0)

        mask = torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]
        return mask

    def _sample_theta_and_magnitude(self, idx):
        if self.base_angle_mode == "fixed_right":
            theta = 0.0
        elif self.base_angle_mode == "fixed_left":
            theta = np.pi
        elif self.base_angle_mode == "fixed_up":
            theta = -np.pi / 2
        elif self.base_angle_mode == "fixed_down":
            theta = np.pi / 2
        elif self.base_angle_mode == "random":
            theta = np.random.uniform(0, 2 * np.pi)
        else:
            rng = np.random.RandomState(idx)
            theta = rng.uniform(0, 2 * np.pi)

        if self.base_angle_mode == "random":
            mag = np.random.uniform(self.base_mag_min, self.base_mag_max)
        else:
            rng = np.random.RandomState(idx + 12345)
            mag = rng.uniform(self.base_mag_min, self.base_mag_max)

        return theta, mag

    def _build_base_condition_from_mask(self, mask, idx):
        """
        mask: [1,H,W] in [0,1]
        return: [3,H,W] = [ux, uy, mag]
        """
        theta, base_mag = self._sample_theta_and_magnitude(idx)

        mask_np = mask.numpy()  # [1,H,W]

        ux = np.cos(theta) * mask_np
        uy = np.sin(theta) * mask_np
        mag = base_mag * mask_np

        cond = np.concatenate([ux, uy, mag], axis=0).astype(np.float32)  # [3,H,W]
        return torch.from_numpy(cond)

    def _random_crop(self, blur, sharp, cond, mask, crop_size):
        _, h, w = blur.shape
        if h < crop_size or w < crop_size:
            return blur, sharp, cond, mask

        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)

        blur = blur[:, top:top + crop_size, left:left + crop_size]
        sharp = sharp[:, top:top + crop_size, left:left + crop_size]
        cond = cond[:, top:top + crop_size, left:left + crop_size]
        mask = mask[:, top:top + crop_size, left:left + crop_size]
        return blur, sharp, cond, mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        blur = self._load_img(sample["blur_path"])
        sharp = self._load_img(sample["sharp_path"])

        _, h, w = sharp.shape
        mask = self._load_mask(sample["mask_path"], size_hw=(h, w))

        # 由 mask 伪造基础 blur condition，字段名仍叫 flow，保持后续接口不变
        flow = self._build_base_condition_from_mask(mask, idx)

        if self.crop_size is not None and self.mode == "train":
            blur, sharp, flow, mask = self._random_crop(blur, sharp, flow, mask, self.crop_size)

        return {
            "blur": blur,
            "sharp": sharp,
            "flow": flow,
            "mask": mask,
            "rel_dir": sample["rel_dir"],
            "stem": sample["stem"],
        }


def masked_flow(flow, mask):
    """
    flow: [B,3,H,W] or [3,H,W]
    mask: [B,1,H,W] or [1,H,W]
    """
    return flow * mask


def valid(model, dataloader, sample_timesteps, device, valid_iters=None, title="None"):
    psnr_func = pyiqa.create_metric('psnr', test_y_channel=False, color_space='rgb').to(device)
    lpips_func = pyiqa.create_metric('lpips').to(device)
    niqe_func = pyiqa.create_metric('niqe').to(device)

    total_val_psnr = AverageMeter()
    total_val_lpips = AverageMeter()
    total_val_niqe = AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        model.eval()
        tq = dataloader if valid_iters is None else islice(dataloader, valid_iters)

        for sample in tq:
            blur = sample['blur'].to(device)
            sharp = sample['sharp'].to(device)
            flow = sample['flow'].to(device)
            mask = sample['mask'].to(device)

            # 验证阶段也保持 mask 约束
            flow = masked_flow(flow, mask)
            condition = torch.cat([sharp, flow], dim=1)

            if args.model == "DDIM":
                output = model.sample(
                    condition=condition,
                    sample_timesteps=sample_timesteps,
                    device=device,
                    tqdm_visible=False
                )
            elif args.model == "DDPM":
                output = model.sample(
                    condition=condition,
                    device=device,
                    tqdm_visible=True
                )
            else:
                raise ValueError(f"Unsupported model {args.model}")

            output = output.clamp(-0.5, 0.5)

            psnr = torch.mean(psnr_func(output.detach(), blur.detach())).item()
            lpips = torch.mean(lpips_func(output.detach(), blur.detach())).item()
            niqe = torch.mean(niqe_func(output.detach())).item()

            total_val_psnr.update(psnr)
            total_val_lpips.update(lpips)
            total_val_niqe.update(niqe)

            if hasattr(tq, "set_postfix"):
                tq.set_postfix(
                    LPIPS=total_val_lpips.avg,
                    PSNR=total_val_psnr.avg,
                    NIQE=total_val_niqe.avg
                )

    elapsed_time = time.time() - start_time
    time_obj = datetime.timedelta(seconds=elapsed_time)
    time_str = str(time_obj).split(".")[0]

    logging.info(f"-----------EVAL------------")
    logging.info(f"Title : {title}")
    logging.info(f"sample_timesteps : {sample_timesteps}")
    logging.info(f"The program's running time is (h:m:s) : {time_str}")
    logging.info(
        f"PSNR : {total_val_psnr.avg:.4f}, "
        f"LPIPS : {total_val_lpips.avg:.4f}, "
        f"NIQE : {total_val_niqe.avg:.4f}"
    )


def val_save_image(model, dir_path, dataset, sample_timesteps, val_num=3, val_idxs=None):
    dir_path = os.path.join(dir_path, "images")
    os.makedirs(dir_path, exist_ok=True)

    with torch.no_grad():
        model.eval()

        if val_idxs is None:
            val_idxs = random.sample(range(0, len(dataset)), min(val_num, len(dataset)))

        for i, idx in enumerate(val_idxs):
            print(i)
            sample = dataset[idx]

            save_sharp_path = os.path.join(dir_path, 'sharp')
            os.makedirs(save_sharp_path, exist_ok=True)
            save_sharp_image_path = os.path.join(save_sharp_path, f'{idx:05d}.png')
            save_image(sample['sharp'].cpu() + 0.5, save_sharp_image_path)

            save_blur_path = os.path.join(dir_path, 'blur')
            os.makedirs(save_blur_path, exist_ok=True)
            save_blur_image_path = os.path.join(save_blur_path, f'{idx:05d}.png')
            save_image(sample['blur'].cpu() + 0.5, save_blur_image_path)

            sharp = sample['sharp'].unsqueeze(0).to(device)
            flow = sample['flow'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)

            flow = masked_flow(flow, mask)
            condition = torch.cat([sharp, flow], dim=1)

            if args.model == "DDIM":
                output = model.sample(
                    condition=condition,
                    sample_timesteps=sample_timesteps,
                    device=device,
                    tqdm_visible=True
                )
            elif args.model == "DDPM":
                output = model.sample(
                    condition=condition,
                    device=device,
                    tqdm_visible=True
                )
            else:
                raise ValueError(f"Unsupported model {args.model}")

            output = output.clamp(-0.5, 0.5)

            save_dir_path = os.path.join(dir_path, 'output')
            os.makedirs(save_dir_path, exist_ok=True)
            save_img_path = os.path.join(save_dir_path, f'{idx:05d}.png')
            output_cv = tensor2cv(output + 0.5)
            cv2.imwrite(save_img_path, output_cv)

            # 可视化当前基础 condition（mask 后）
            flow_np = flow.squeeze(0).cpu().numpy().transpose((1, 2, 0))
            flow_x = flow_np[:, :, 0] * flow_np[:, :, 2]
            flow_y = flow_np[:, :, 1] * flow_np[:, :, 2]
            optical_flow = np.stack((flow_x, flow_y), axis=-1)

            flo = flow_to_image(optical_flow, norm=1)

            flow_dir_path = os.path.join(dir_path, 'flow')
            os.makedirs(flow_dir_path, exist_ok=True)
            flow_img_path = os.path.join(flow_dir_path, f'{idx:05d}.png')
            cv2.imwrite(flow_img_path, flo[:, :, [2, 1, 0]])


def generate_dataset(model, dir_path, dataset, sample_timesteps, strategySetting, generate_num=5, save_npy=False):
    """
    生成多种 reblur 图像
    输出目录保留原层级结构：
      dir_path/
        sharp/<rel_dir>/<stem>/sharp.png
        blur/<rel_dir>/<stem>/00000.png
        condition/<rel_dir>/<stem>/00000.npy
    """
    sharp_path = os.path.join(dir_path, "sharp")
    blur_path = os.path.join(dir_path, "blur")
    condition_path = os.path.join(dir_path, "condition")

    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(sharp_path, exist_ok=True)
    os.makedirs(blur_path, exist_ok=True)
    os.makedirs(condition_path, exist_ok=True)

    if 'TURN' not in strategySetting:
        strategy = strategySetting[:]
        strategy_list = None
    else:
        strategy_list = strategySetting[:]
        strategy_list.remove('TURN')
        if 'FIXED' in strategySetting:
            strategy_list.remove("FIXED")
        strategy = None

    with torch.no_grad():
        model.eval()
        for idx in range(len(dataset)):
            print(f"Processing {idx + 1}/{len(dataset)}")
            sample = dataset[idx]

            rel_dir = sample["rel_dir"]
            stem = sample["stem"]

            sharp_idx_path = os.path.join(sharp_path, rel_dir, stem)
            blur_idx_path = os.path.join(blur_path, rel_dir, stem)
            condition_idx_path = os.path.join(condition_path, rel_dir, stem)

            os.makedirs(sharp_idx_path, exist_ok=True)
            os.makedirs(blur_idx_path, exist_ok=True)
            if save_npy:
                os.makedirs(condition_idx_path, exist_ok=True)

            save_sharp_image_path = os.path.join(sharp_idx_path, 'sharp.png')
            save_image(sample['sharp'].cpu() + 0.5, save_sharp_image_path)

            sharp = sample['sharp'].unsqueeze(0).to(device)
            flow = sample['flow'].clone().unsqueeze(0).to(device)   # 基础伪 condition
            mask = sample['mask'].unsqueeze(0).to(device)

            change_base = 0
            if 'FIXED' in strategySetting:
                change_base = random.randint(0, 100)

            for index in range(generate_num):
                choice_num = None

                if strategy_list is not None:
                    if 'FIXED' in strategySetting:
                        choice_num = index
                    strategy = [strategy_list[(idx + index) % len(strategy_list)]]

                # 仍然使用原来的 select_condition_strategy
                new_flow = select_condition_strategy(
                    flow,
                    strategy=strategy,
                    choice_num=choice_num,
                    change_base=change_base
                )

                # 再根据 mask 把不想增加模糊的区域置 0
                new_flow = masked_flow(new_flow, mask)

                condition = torch.cat([sharp, new_flow], dim=1)

                if args.model == "DDIM":
                    output = model.sample(
                        condition=condition,
                        sample_timesteps=sample_timesteps,
                        device=device,
                        tqdm_visible=False
                    )
                elif args.model == "DDPM":
                    output = model.sample(
                        condition=condition,
                        device=device,
                        tqdm_visible=False
                    )
                else:
                    raise ValueError(f"Unsupported model {args.model}")

                output = output.clamp(-0.5, 0.5)

                save_img_path = os.path.join(blur_idx_path, f'{index:05d}.png')
                output_cv = tensor2cv(output + 0.5)
                cv2.imwrite(save_img_path, output_cv)

                if save_npy:
                    condition_np = new_flow.squeeze(0).cpu().numpy()
                    save_npy_path = os.path.join(condition_idx_path, f'{index:05d}.npy')
                    np.save(save_npy_path, condition_np)


def generate_linear_schedule(T, beta_1, beta_T):
    return torch.linspace(beta_1, beta_T, T).double()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--data_path", default='./dataset/reloblur', type=str)
    parser.add_argument("--mask_path", default="./dataset/reloblur/mask", type=str)
    parser.add_argument("--dir_path", default='./dataset/reloblur_aug_fakeflow', type=str)
    parser.add_argument("--model_path", default='./weights/ID_Blau.pth', type=str)

    parser.add_argument("--model", default='DDIM', type=str)
    parser.add_argument("--title", default='None', type=str)
    parser.add_argument(
        "--type",
        default='generate_dataset',
        type=str,
        choices=['generate_dataset']
    )
    parser.add_argument("--dataset", default='train', type=str, choices=['train', 'test'])
    parser.add_argument("--val_num", default=10, type=int)
    parser.add_argument(
        "--strategy",
        default=['ALLM', 'ALLO'],
        type=str,
        choices=['O', 'M10', 'M20', 'M30', 'M40', 'M60', 'M80', 'ALLM', 'ALLO', 'RO', '30O', '60O', 'FIXED', 'TURN'],
        nargs='+'
    )
    parser.add_argument("--sample_timesteps", default=20, type=int)
    parser.add_argument("--generate_num", default=5, type=int)
    parser.add_argument("--valid_iters", default=None, type=int)
    parser.add_argument("--crop_size", default=None, type=int)
    parser.add_argument("--save_npy", action="store_true")
    parser.add_argument("--seed", default=2023, type=int)

    # 基础伪 condition 的生成参数
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--mask_blur_ksize", default=21, type=int)
    parser.add_argument("--base_mag_min", default=0.30, type=float)
    parser.add_argument("--base_mag_max", default=0.60, type=float)
    parser.add_argument(
        "--base_angle_mode",
        default="random",
        type=str,
        choices=["deterministic_by_index", "random", "fixed_right", "fixed_left", "fixed_up", "fixed_down"]
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    same_seed(args.seed)

    load_model_state = torch.load(args.model_path)
    model_args = load_model_state['args']

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path, exist_ok=True)

    val_idxs = None

    if args.dataset == "train":
        dataset = PairMaskConditionLoader(
            data_path=args.data_path,
            mask_path=args.mask_path,
            mode="train",
            crop_size=args.crop_size,
            invert_mask=args.invert_mask,
            mask_blur_ksize=args.mask_blur_ksize,
            base_mag_min=args.base_mag_min,
            base_mag_max=args.base_mag_max,
            base_angle_mode=args.base_angle_mode
        )
    elif args.dataset == "test":
        dataset = PairMaskConditionLoader(
            data_path=args.data_path,
            mask_path=args.mask_path,
            mode="test",
            crop_size=args.crop_size,
            invert_mask=args.invert_mask,
            mask_blur_ksize=args.mask_blur_ksize,
            base_mag_min=args.base_mag_min,
            base_mag_max=args.base_mag_max,
            base_angle_mode=args.base_angle_mode
        )
    else:
        raise ValueError("Invalid dataset type (only train and test)")

    beta = generate_linear_schedule(
        model_args.num_timesteps, model_args.beta_1, model_args.beta_T
    )

    model_UNet = UNet(
        channel_mults=model_args.channel_mults,
        base_channels=model_args.base_channels,
        time_dim=model_args.time_dim,
        dropout=model_args.dropout
    ).to(device)

    if args.model == "DDIM":
        diffusionModel = DDIM(model_UNet, img_channels=9, betas=beta).to(device)
    elif args.model == "DDPM":
        diffusionModel = DDPM(model_UNet, img_channels=9, betas=beta).to(device)
    else:
        raise ValueError(f"model not supported {args.model}")

    if 'model_state' in load_model_state.keys():
        diffusionModel.load_state_dict(load_model_state["model_state"])
    else:
        diffusionModel.load_state_dict(load_model_state)

    print("device:", device)
    print(f'args: {args}')
    print(f'model parameters: {count_parameters(diffusionModel)}')

    if args.type == 'generate_dataset':
        print(f'strategy: {args.strategy}')
        generate_dataset(
            diffusionModel,
            args.dir_path,
            dataset,
            sample_timesteps=args.sample_timesteps,
            generate_num=args.generate_num,
            strategySetting=args.strategy,
            save_npy=args.save_npy
        )

    if args.type in pyiqa.list_models():
        logging.basicConfig(
            filename=os.path.join(args.dir_path, 'eval.log'),
            format='%(asctime)s | %(levelname)s : %(message)s',
            encoding='utf-8',
            level=logging.INFO
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s : %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=False
        )

        valid(
            diffusionModel,
            dataloader,
            sample_timesteps=args.sample_timesteps,
            device=device,
            valid_iters=args.valid_iters,
            title=args.title
        )

    elif args.type == "image":
        val_save_image(
            diffusionModel,
            args.dir_path,
            dataset,
            sample_timesteps=args.sample_timesteps,
            val_num=args.val_num,
            val_idxs=val_idxs
        )