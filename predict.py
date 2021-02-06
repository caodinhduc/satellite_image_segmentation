import argparse
import logging
import os

import numpy as np
import torch
import time
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    """ Predict single image"""
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/best_checkpoint.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', default='test.tif',   metavar='INPUT',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', default='test_out.tif', metavar='INPUT',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    start_time = time.time()

    net = UNet(n_channels=3, n_classes=1)
    print("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    print("Model loaded !")

    print('model loaded takes {} seconds'.format(time.time() - start_time))
    start_time = time.time()

    img = Image.open(args.input)

    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=args.scale,
                        out_threshold=args.mask_threshold,
                        device=device)

    result = mask_to_image(mask)
    print('model inference takes {} seconds'.format(time.time() - start_time))


    result.save(args.output)

    if args.viz:
        logging.info("Visualizing results for image {}, close to continue ...".format(args.input))
        plot_img_and_mask(img, mask)
