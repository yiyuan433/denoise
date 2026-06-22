"""
数据增强工具 - 针对低SNR数据
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DataAugmentation2D:
    """2D数据增强"""
    
    def __init__(self, config):
        self.config = config
    
    def __call__(self, patch):
        """应用随机增强"""
        if isinstance(patch, torch.Tensor):
            patch = patch.numpy()
        
        # 随机水平翻转
        if self.config.get('horizontal_flip', False) and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=2).copy()
        
        # 随机垂直翻转
        if self.config.get('vertical_flip', False) and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1).copy()
        
        # 随机旋转
        if self.config.get('rotation', False):
            rotations = self.config.get('rotation', [0, 90, 180, 270])
            if isinstance(rotations, bool):
                rotations = [0, 90, 180, 270]
            angle = np.random.choice(rotations)
            k = angle // 90
            if k > 0:
                patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
        
        # 随机转置
        if self.config.get('transpose', False) and np.random.rand() > 0.5:
            patch = np.transpose(patch, (0, 2, 1)).copy()
        
        # 噪声增强（添加额外弱噪声）
        if self.config.get('noise_augment', False) and np.random.rand() > 0.5:
            noise_range = self.config.get('noise_range', (0.05, 0.15))
            noise_level = np.random.uniform(*noise_range)
            noise = np.random.randn(*patch.shape) * noise_level * np.std(patch)
            patch = patch + noise
        
        # Cutout（随机遮挡）
        if self.config.get('cutout', False) and np.random.rand() > 0.5:
            patch = self.apply_cutout(patch)
        
        return torch.from_numpy(patch).float()
    
    def apply_cutout(self, patch):
        """应用cutout增强"""
        ratio = self.config.get('cutout_ratio', 0.1)
        _, h, w = patch.shape
        
        cut_h = int(h * ratio)
        cut_w = int(w * ratio)
        
        # 随机位置
        y = np.random.randint(0, h - cut_h + 1)
        x = np.random.randint(0, w - cut_w + 1)
        
        # 遮挡（填充为均值）
        patch[:, y:y+cut_h, x:x+cut_w] = np.mean(patch)
        
        return patch


class AugmentedDataset(Dataset):
    """带增强的数据集"""
    
    def __init__(self, patches, augmentation_config, add_noise=True):
        self.patches = patches
        self.augmentation = DataAugmentation2D(augmentation_config)
        self.add_noise = add_noise
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        # 应用增强
        patch_aug = self.augmentation(patch)
        
        # Clean版本（作为target）
        clean = patch_aug.clone()
        
        # 添加噪声（作为input）
        if self.add_noise:
            noise_std = np.random.uniform(0.1, 0.3) * torch.std(patch_aug)
            noisy = patch_aug + torch.randn_like(patch_aug) * noise_std
        else:
            noisy = patch_aug
        
        return noisy, clean


def create_patches_2d(data, patch_size=(48, 48), stride=(12, 12)):
    """创建2D patches"""
    h, w = data.shape
    ph, pw = patch_size
    sh, sw = stride
    
    patches = []
    
    for i in range(0, h - ph + 1, sh):
        for j in range(0, w - pw + 1, sw):
            patch = data[i:i+ph, j:j+pw]
            # 添加channel维度
            patch = patch[np.newaxis, :, :]
            patches.append(patch)
    
    patches = np.array(patches)
    print(f"创建了 {len(patches)} 个 patch")
    
    return torch.from_numpy(patches).float()


def visualize_augmentation(patch, config, num_samples=8):
    """可视化数据增强效果"""
    import matplotlib.pyplot as plt
    
    augmentation = DataAugmentation2D(config)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 原始patch
    axes[0].imshow(patch[0], cmap='seismic', aspect='auto')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 增强后的patch
    for i in range(1, num_samples):
        aug_patch = augmentation(torch.from_numpy(patch))
        axes[i].imshow(aug_patch[0], cmap='seismic', aspect='auto')
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # 测试数据增强
    print("测试数据增强...")
    
    # 加载数据
    available_data = sorted([
        os.path.join(PATHS['data'], name)
        for name in os.listdir(PATHS['data'])
        if name.endswith('.npy')
    ])
    data_path = available_data[0]
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")
    
    # 创建patches
    patches = create_patches_2d(data, patch_size=(48, 48), stride=(12, 12))
    print(f"Patches形状: {patches.shape}")
    
    # 测试增强
    aug_config = {
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation': [0, 90, 180, 270],
        'transpose': True,
        'noise_augment': True,
        'noise_range': (0.05, 0.15),
        'cutout': True,
        'cutout_ratio': 0.1,
    }
    
    # 可视化
    fig = visualize_augmentation(patches[0:1].numpy(), aug_config)
    
    save_path = r'f:\项目（老师）\denoise\result\das_denoise_3\new\figures\augmentation_examples.png'
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存: {save_path}")
    
    plt.show()
