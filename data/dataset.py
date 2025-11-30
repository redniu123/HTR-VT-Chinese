import numpy as np
import torch
import skimage
import os
import itertools
from PIL import Image
from torch.utils.data import Dataset
try:
    from utils import my_utils as utils
except ImportError:
    from utils import utils
from data import transform as transform
from torchvision.transforms import ColorJitter
from tqdm import tqdm

def SameTrCollate(batch, args):
    """
    Collate function with Data Augmentation
    Expects input 'images' as list of float32 numpy arrays [C, H, W] in range [0, 1]
    """
    # 1. 解包 Batch
    raw_images, labels = zip(*batch)

    # 2. 将 Numpy 数组转换为 PIL Images (为了进行数据增强)
    pil_images = []
    for img in raw_images:
        # img shape: [C, H, W] (例如 [1, 64, 512])
        
        # 取出图像数据 (假设是灰度图 C=1)
        if img.ndim == 3:
            img_data = img[0]
        else:
            img_data = img
            
        # 还原数值范围 (0.0-1.0 -> 0-255) 并转为 uint8
        img_uint8 = (img_data * 255.0).astype(np.uint8)
        
        # 转为 PIL Image ('L' 表示灰度模式)
        pil_img = Image.fromarray(img_uint8, mode='L')
        pil_images.append(pil_img)

    # 3. 应用数据增强 (Data Augmentation)
    # --- [关键开启] 几何变换 (防止过拟合) ---
    if np.random.rand() < 1:
        try:
            pil_images = [transform.RandomTransform(args.proj)(image) for image in pil_images]
        except Exception:
            pass # 如果变换失败，保持原图

    # --- [可选开启] 腐蚀膨胀 (稍微耗时，如果训练太慢可注释掉) ---
    if np.random.rand() < 0.5:
        kernel_h = utils.randint(1, args.dila_ero_max_kernel + 1)
        kernel_w = utils.randint(1, args.dila_ero_max_kernel + 1)
        if utils.randint(0, 2) == 0:
            pil_images = [transform.Erosion((kernel_w, kernel_h), args.dila_ero_iter)(image) for image in pil_images]
        else:
            pil_images = [transform.Dilation((kernel_w, kernel_h), args.dila_ero_iter)(image) for image in pil_images]

    # --- [开启] 颜色抖动 (速度快) ---
    if np.random.rand() < 0.5:
        pil_images = [ColorJitter(args.jitter_brightness, args.jitter_contrast, args.jitter_saturation,
                              args.jitter_hue)(image) for image in pil_images]

    # 4. 转回 Tensor 并归一化
    # uint8 [0, 255] -> float32 [0.0, 1.0]
    image_tensors = [torch.from_numpy(np.array(image, copy=True)) for image in pil_images]
    
    # Stack: [B, H, W]
    image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
    
    # Add Channel & Normalize: [B, 1, H, W]
    image_tensors = image_tensors.unsqueeze(1).float()
    image_tensors = image_tensors / 255.
    
    return image_tensors, labels


class myLoadDS(Dataset):
    def __init__(self, flist, dpath, img_size=[512, 32], ralph=None, fmin=True, mln=None):
        self.fns = get_files(flist, dpath)
        self.tlbls = get_labels(self.fns)
        self.img_size = img_size

        if ralph == None:
            alph = get_alphabet(self.tlbls)
            self.ralph = dict(zip(alph.values(), alph.keys()))
            self.alph = alph
        else:
            self.ralph = ralph

        # --- [RAM Cache] 预加载所有图片到内存 ---
        print(f"🚀 Pre-loading {len(self.fns)} images to RAM...")
        self.cached_images = []
        
        for fname in tqdm(self.fns, desc="Caching"):
            try:
                # 1. 读取
                img = Image.open(fname).convert('L')
                # 2. Resize 和 Pad
                img_np = npThum(np.array(img), img_size[0], img_size[1])
                
                # 3. 统一处理成标准 numpy 格式存入内存 (uint8 省内存)
                h, w = img_np.shape[:2]
                pad_img = np.ones((img_size[1], img_size[0]), dtype=np.uint8) * 255 # 白底
                pad_img[:h, :w] = img_np
                
                self.cached_images.append(pad_img)
                
            except Exception as e:
                # 容错：给一张全白图
                self.cached_images.append(np.ones((img_size[1], img_size[0]), dtype=np.uint8) * 255)
        
        print("✅ All images loaded to RAM!")

        if mln != None:
            filt = [len(x) <= mln if fmin else len(x) >= mln for x in self.tlbls]
            self.tlbls = np.asarray(self.tlbls)[filt].tolist()
            self.fns = np.asarray(self.fns)[filt].tolist()

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, index):
        # 1. 从内存获取缓存的 uint8 数据 [H, W]
        img_data_uint8 = self.cached_images[index] 
        
        # 2. 转为 float32 并归一化到 [0, 1]
        # 这是为了解决 LayerNorm 不支持 Byte 的报错
        img_data_float = img_data_uint8.astype(np.float32) / 255.0
        
        # 3. 增加 Channel 维度 [H, W] -> [H, W, 1]
        if img_data_float.ndim == 2:
            img_data_float = img_data_float[:, :, np.newaxis] 
            
        # 4. 转置为 PyTorch 格式 [C, H, W]
        timgs = img_data_float.transpose((2, 0, 1)) 
        
        return (timgs, self.tlbls[index])


def get_files(nfile, dpath):
    fnames = open(nfile, 'r').readlines()
    fnames = [os.path.join(dpath, x.strip()) for x in fnames] # 优化路径拼接
    return fnames


def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]
    y = min(int(y * max_h / x), max_w)
    x = max_h
    img = np.array(Image.fromarray(img).resize((y, x)))
    return img


def get_images(fname, max_w=500, max_h=500, nch=1): 
    # 这个函数现在其实已经不被使用了，因为逻辑移到了 __init__ 里
    # 但为了兼容性保留
    try:
        image_data = np.array(Image.open(fname).convert('L'))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        h, w = np.shape(image_data)[:2]
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)

        if nch == 3 and image_data.shape[2] != 3:
            image_data = np.tile(image_data, 3)

        image_data = np.pad(image_data, ((0, 0), (0, max_w - np.shape(image_data)[1]), (0, 0)), mode='constant', constant_values=(1.0))

    except IOError as e:
        print('Could not read:', fname, ':', e)
        return np.zeros((max_h, max_w, nch))

    return image_data


def get_labels(fnames):
    labels = []
    print(f"正在读取 {len(fnames)} 个标签文件...") 

    for id, image_file in enumerate(tqdm(fnames, desc="Loading Labels")):
        fn = os.path.splitext(image_file)[0] + '.txt'
        try:
            lbl = open(fn, 'r', encoding='utf-8').read()
        except Exception:
            lbl = "" 

        lbl = ' '.join(lbl.split())
        labels.append(lbl)

    return labels


def get_alphabet(labels):
    coll = ''.join(labels)
    unq = sorted(list(set(coll)))
    unq = [''.join(i) for i in itertools.product(unq, repeat=1)]
    alph = dict(zip(unq, range(len(unq))))
    return alph


def cycle_dpp(iterable):
    epoch = 0
    iterable.sampler.set_epoch(epoch)
    while True:
        for x in iterable:
            yield x
        epoch += 1
        iterable.sampler.set_epoch(epoch)


def cycle_data(iterable):
    while True:
        for x in iterable:
            yield x