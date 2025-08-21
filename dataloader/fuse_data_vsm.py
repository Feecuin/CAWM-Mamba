import torch.utils.data
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import numpy as np
from dataloader import transforms as T


def _imread(path):
    im_cv = Image.open(path).convert('L')
    # im_cv = cv2.imread(str(path), flags)
    im_cv = im_cv.resize((600,400), Image.ANTIALIAS)
    assert im_cv is not None, f"Image {str(path)} is invalid."
    # im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
    tran = transforms.ToTensor()
    im_ts = tran(im_cv) / 255.
    return im_ts




class GetDataset_type3(torch.utils.data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None, gt_path=None, gt_ir_path=None, img_size=None):
        super(GetDataset_type3, self).__init__()

        if split == 'train':
            data_dir_ir = ir_path
            data_dir_vis = vi_path
            data_dir_gt = gt_path
            data_dir_gt_ir = gt_ir_path

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_gt, self.filenames_gt = prepare_data_path(data_dir_gt)
            self.filepath_gt_ir, self.filenames_gt_ir = prepare_data_path(data_dir_gt_ir)


            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir), len(self.filenames_gt),
                               len(self.filenames_gt_ir))

            self.transform = T.Compose([T.RandomCrop(img_size),
                                    T.RandomHorizontalFlip(0.5),
                                    T.RandomVerticalFlip(0.5),
                                    T.ToTensor()])

    def __getitem__(self, index):
        if self.split=='train':
            # print('-----------')
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            gt_path = self.filepath_gt[index]
            gt_ir_path = self.filepath_gt_ir[index]


            image_vis = Image.open(vis_path).convert(mode='RGB')
            image_ir = Image.open(ir_path).convert(mode='RGB')
            image_gt = Image.open(gt_path).convert(mode='RGB')
            image_gt_ir = Image.open(gt_ir_path).convert(mode='RGB')

            image_ir, image_vis, image_gt, image_gt_ir = self.transform(image_ir, image_vis, image_gt, image_gt_ir)

            return (
                image_ir,
                image_vis,
                image_gt,
                image_gt_ir,
            )

    def __len__(self):
        # print(self.length)
        return self.length

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.txt"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.npy"))))
    data.sort()
    filenames.sort()
    return data, filenames

def prepare_clip_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.txt"))
    data.sort()
    filenames.sort()
    return data, filenames

def prepare_blip_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.npy"))
    data.sort()
    filenames.sort()
    return data, filenames

if __name__ == "__main__":
    train_dataset = GetDataset_type3('train')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
    )
    i = 0
    for vi, ir in train_loader:
        i += 1

        ir = ir.permute(0, 2, 3, 1)
        vi = vi.permute(0, 2, 3, 1)
        ir = torch.squeeze(ir, 0)
        vi = torch.squeeze(vi, 0)

        ir = ir.numpy()
        vi = vi.numpy()
        ir = (ir * 255).astype(np.uint8)
        vi = (vi * 255).astype(np.uint8)