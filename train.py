import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from PIL import Image
from tqdm import tqdm

from eval import eval_net
from unet import UNet, UNetResNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from predict import predict_img, mask_to_image

torch.cuda.empty_cache()
dir_img = 'data/train/imgs/'
dir_mask = 'data/train/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=5,
              batch_size=2,
              lr=0.0001,
              val_percent=0.2,
              save_cp=True,
              img_scale=1):

    # Init dataset and train/test split
    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    # Call DataLoader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # Writer to tensorboard
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
            Images scaling:  {img_scale}
        ''')

    # Init optimizer and define lr_scheduler
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    
    # In this version, we use BCEWithLogitsLoss 
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
              
                # Set-up device
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                # Forward
                masks_pred = net(imgs)
                
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1


                if global_step % (n_train // (2 * batch_size)) == 0:
                # if global_step % 100 == 0:
                    # Track weight and gradient
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    
                    val_score, val_score_iou = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    # Visualize val scores
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    logging.info('Validation IoU Coeff: {}'.format(val_score_iou))
                    writer.add_scalar('Dice/test', val_score, global_step)
                    writer.add_scalar('IoU/test', val_score_iou, global_step)
                    writer.add_images('images', imgs, global_step)

                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        # test with sample images
        test_folder = 'test/test_epoch_{}_new'.format(epoch)
        os.makedirs(test_folder)
        dirs = os.listdir('test/test_set')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for file in dirs:
            img = Image.open(os.path.join('test/test_set', file))
            mask = predict_img(net=net,
                    full_img=img,
                    device=device)

            result = mask_to_image(mask)
            result.save(os.path.join(test_folder, file))


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':

    # Config logging and device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    """ n_channels=3 for RGB images
        n_classes is the number of probabilities you want to get per pixel
        For 1 class and background, use n_classes=1 """
 
    net = UNet(n_channels=3, n_classes=1, bilinear=True)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)