import os
import random
import torch
from pathlib import Path
from PIL import Image
class StreetObstacleSequences(torch.utils.data.Dataset):
    train_id_in = 0
    train_id_out = 254

    def __init__(self, root, split, test_set_size=20, transform=None):
        self.transform = transform
        self.test_set_size = test_set_size

        self.images = []
        self.targets = []

        self.image_root = Path(root) / "raw_data"
        self.target_root = Path(root) / "semantic_ood"

        print(os.getcwd())
        sequences = os.listdir(self.target_root)

        for sequence in sequences:
            files = os.listdir(self.target_root / sequence)
            for file in files:
                target_name = sequence + "/" + file
                img_nr = file.split("_")[0]
                img_name = sequence + "/" + img_nr + "_raw_data.jpg"
                self.images.append(img_name)
                self.targets.append(target_name)

        indices = random.sample(range(len(self.images)), self.test_set_size)
        self.images = [self.images[i] for i in indices]
        self.targets = [self.targets[i] for i in indices]
        # manually set images and targets for validation across approaches
        self.images = ['sequence_018/000072_raw_data.jpg', 'sequence_002/000184_raw_data.jpg',
                       'sequence_003/000432_raw_data.jpg', 'sequence_007/000240_raw_data.jpg',
                       'sequence_014/000056_raw_data.jpg', 'sequence_004/000408_raw_data.jpg',
                       'sequence_019/000224_raw_data.jpg', 'sequence_006/000120_raw_data.jpg',
                       'sequence_003/000208_raw_data.jpg', 'sequence_010/000280_raw_data.jpg',
                       'sequence_003/000280_raw_data.jpg', 'sequence_004/000312_raw_data.jpg',
                       'sequence_009/000352_raw_data.jpg', 'sequence_005/000008_raw_data.jpg',
                       'sequence_008/000120_raw_data.jpg', 'sequence_020/000072_raw_data.jpg',
                       'sequence_005/000088_raw_data.jpg', 'sequence_020/000208_raw_data.jpg',
                       'sequence_003/000184_raw_data.jpg', 'sequence_001/000232_raw_data.jpg',
                       'sequence_012/000128_raw_data.jpg', 'sequence_012/000152_raw_data.jpg',
                       'sequence_003/000416_raw_data.jpg', 'sequence_015/000064_raw_data.jpg',
                       'sequence_002/000024_raw_data.jpg', 'sequence_013/000240_raw_data.jpg',
                       'sequence_018/000080_raw_data.jpg', 'sequence_001/000176_raw_data.jpg',
                       'sequence_010/000320_raw_data.jpg', 'sequence_016/000136_raw_data.jpg']
        self.targets = ['sequence_018/000072_semantic_ood.png', 'sequence_002/000184_semantic_ood.png',
                        'sequence_003/000432_semantic_ood.png', 'sequence_007/000240_semantic_ood.png',
                        'sequence_014/000056_semantic_ood.png', 'sequence_004/000408_semantic_ood.png',
                        'sequence_019/000224_semantic_ood.png', 'sequence_006/000120_semantic_ood.png',
                        'sequence_003/000208_semantic_ood.png', 'sequence_010/000280_semantic_ood.png',
                        'sequence_003/000280_semantic_ood.png', 'sequence_004/000312_semantic_ood.png',
                        'sequence_009/000352_semantic_ood.png', 'sequence_005/000008_semantic_ood.png',
                        'sequence_008/000120_semantic_ood.png', 'sequence_020/000072_semantic_ood.png',
                        'sequence_005/000088_semantic_ood.png', 'sequence_020/000208_semantic_ood.png',
                        'sequence_003/000184_semantic_ood.png', 'sequence_001/000232_semantic_ood.png',
                        'sequence_012/000128_semantic_ood.png', 'sequence_012/000152_semantic_ood.png',
                        'sequence_003/000416_semantic_ood.png', 'sequence_015/000064_semantic_ood.png',
                        'sequence_002/000024_semantic_ood.png', 'sequence_013/000240_semantic_ood.png',
                        'sequence_018/000080_semantic_ood.png', 'sequence_001/000176_semantic_ood.png',
                        'sequence_010/000320_semantic_ood.png', 'sequence_016/000136_semantic_ood.png']

    def __len__(self):
        return len(self.targets)


    def __getitem__(self, i):
        image_path = self.image_root / self.images[i]
        image = Image.open(image_path).convert('RGB')
        target_path = self.target_root / self.targets[i]
        target = Image.open(target_path).convert('L')
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


if __name__ == '__main__':
    ds = StreetObstacleSequences("", "")
