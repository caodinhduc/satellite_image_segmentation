import cv2
import os

imgs_folder = './data/train/imgs/'
imgs_mask_folder = './data/train/masks/'
save_imgs_folder = './data/train_cut/imgs/'
save_imgs_mask_folder = './data/train_cut/masks/'
test_folder = './data/test/images/'
save_test_folder = './data/test_cut/'

# print(os.listdir(imgs_mask_folder))
def cut_img(folder, save_folder, mask = False):
    for img_folder in os.listdir(folder):
        img = cv2.imread(folder + img_folder)
        if mask:
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            (_, img) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        for i in range(10):
            for j in range(10):
                img_i_j = img[i*500:(i+1)*500, j*500:(j+1)*500]
                image_path = os.path.join(save_folder, img_folder.split('.')[0] + '_' + str(i) + '_' +  str(j) + '.tif')
                print(image_path)
                cv2.imwrite(image_path, img_i_j)

#===============================================#
cut_img(test_folder, save_test_folder)
# cut_img(imgs_mask_folder, save_imgs_mask_folder, True)

# img = cv2.imread(imgs_mask_folder + 'austin1.tif')
# grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
# print(blackAndWhiteImage.shape)
            