from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import imgaug.random as iarandom
from scipy.ndimage import gaussian_filter
import cv2, os, json

from pprint import pprint
### JSON

# read 1 json and output [['left',x,y],['left',x,y],...,[0.0,0.0,0.0]]
def read_json(json_path, scale_h, scale_w):
    json_file = os.path.join(json_path)
    f = open(json_file)
    dic = json.load(f)
    nodes = dic['shapes']
    all_points = np.zeros((8,4)).tolist()
    for node in nodes:
        n_name = node['label'].split('_')   # left_1
        if   n_name[0] == 'left' and n_name[1] == '0':
            p = 0
        elif n_name[0] == 'right' and n_name[1] == '0':
            p = 1
        elif n_name[0] == 'joint' and n_name[1] == '0':
            p = 2
        elif n_name[0] == 'em' and n_name[1] == '0':
            p = 3
        elif n_name[0] == 'left' and n_name[1] == '1':
            p = 4
        elif n_name[0] == 'right' and n_name[1] == '1':
            p = 5
        elif n_name[0] == 'joint' and n_name[1] == '1':
            p = 6
        elif n_name[0] == 'em' and n_name[1] == '1':
            p = 7
        else:
            p = -1
        if p >= 0:
            all_points[p][0] = n_name[0]    # for name
            all_points[p][1] = n_name[1]    # for tool id
            x = int(round(node['points'][0][0] * scale_w))
            all_points[p][2] = x
            y = int(round(node['points'][0][1] * scale_h))
            all_points[p][3] = y
    return all_points

def draw_point(rd, feat_num, node, y_bin, y_den):
    bin_layer = np.array(y_bin[:, :, feat_num])
    den_layer = np.array(y_den[:, :, feat_num])

    y_bin[:, :, feat_num] = cv2.circle(bin_layer,
                                      (node[2], node[3]),
                                      rd, 1, cv2.FILLED)
    y_den[:, :, feat_num] = cv2.circle(den_layer,
                                      (node[2], node[3]),
                                      1, 1, cv2.FILLED)

    return y_bin, y_den

def draw_line(rd, feat_num, all_points, y_bin, y_den, j, skip):
    bin_layer = np.array(y_bin[:, :, feat_num])
    den_layer = np.array(y_den[:, :, feat_num])

    if all_points[j+skip][2] != 0.0 and all_points[j+skip][3] != 0.0:
        y_bin[:, :, feat_num] = cv2.line(bin_layer,
                                        (all_points[j][2], all_points[j][3]),
                                        (all_points[j+skip][2], all_points[j+skip][3]),  # joint
                                        1, rd)

        y_den[:, :, feat_num] = cv2.line(den_layer,
                                        (all_points[j][2], all_points[j][3]),
                                        (all_points[j+skip][2], all_points[j+skip][3]),
                                        1, 1)

    return y_bin, y_den


class DataGenerator(Sequence):

    def __init__(self, img_list, json_list, input_shape, batch_size=2,
                 sigma=20, rd=15, shuffle=True, aug_proba=0.3):
        self.img_list = img_list
        self.json_list = json_list
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.sigma = sigma
        self.rd = rd
        self.shuffle = shuffle
        self.aug_proba = aug_proba
        self.on_epoch_end()

    # output the length of the list
    def __len__(self):
        return int(np.floor(len(self.img_list)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_img = [self.img_list[i] for i in indexes]
        batch_json = [self.json_list[j] for j in indexes]
        x, [y_bin, y_den] = self.__data_generation(batch_img,
                                                   batch_json)
        return x, [y_bin, y_den]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_img, batch_json):
        c = self.input_shape[0]
        h = self.input_shape[1]
        w = self.input_shape[2]

        x = np.empty((self.batch_size, h, w, c))
        y_bin = np.empty((self.batch_size, h, w, 7))
        y_den = np.empty((self.batch_size, h, w, 7))
        if self.aug_proba:
            sometimes = lambda aug: iaa.Sometimes(self.aug_proba, aug)
            num_gen = iarandom.RNG(111)
            
            seq_con = iaa.SomeOf(2, [
                sometimes(iaa.Affine(rotate=num_gen.randint(-45, 45),
                                     shear=num_gen.randint(-30, 30))),
                sometimes(iaa.ElasticTransformation(alpha=30)),
                iaa.Flipud(self.aug_proba),
                iaa.Fliplr(self.aug_proba),
                iaa.Cutout(nb_iterations=2, size=0.1, fill_mode="constant",
                           cval=0, fill_per_channel=True)
            ])

            seq_img = iaa.Sequential([
                sometimes(iaa.JpegCompression(compression=(80, 90)))
            ])

        for i, input_packs in enumerate(zip(batch_img, batch_json)):
            img = input_packs[0]
            jsons = input_packs[1]

            img_temp = cv2.imread(img)
            scale_w = w/img_temp.shape[1]
            scale_h = h/img_temp.shape[0]
            img_temp = cv2.resize(img_temp, (w, h))

            all_points = read_json(jsons, scale_h, scale_w)
            y_bin_temp = np.zeros((h, w, 7))
            y_den_temp = np.zeros((h, w, 7))
            # y_bin circle y_den point
            for node in all_points:
                if node[0] == 'left':
                    y_bin_temp, y_den_temp = draw_point(self.rd, 0, node, y_bin_temp, y_den_temp)

                elif node[0] == 'right':
                    y_bin_temp, y_den_temp = draw_point(self.rd, 1, node, y_bin_temp, y_den_temp)

                elif node[0] == 'joint':
                    y_bin_temp, y_den_temp = draw_point(self.rd, 2, node, y_bin_temp, y_den_temp)

                elif node[0] == 'em':
                    y_bin_temp, y_den_temp = draw_point(self.rd, 3, node, y_bin_temp, y_den_temp)


            # y_bin line
            for j, node in enumerate(all_points):
                # left-joint
                if node[0] == 'left':
                    y_bin_temp, y_den_temp = draw_line(self.rd, 4, all_points, y_bin_temp, y_den_temp, j, skip=2)

                # right-joint
                elif node[0] == 'right':
                    y_bin_temp, y_den_temp = draw_line(self.rd, 5, all_points, y_bin_temp, y_den_temp, j, skip=1)

                # joint-em
                elif node[0] == 'joint':
                    y_bin_temp, y_den_temp = draw_line(self.rd, 6, all_points, y_bin_temp, y_den_temp, j, skip=1)
            
            # gaussian filter for each layer
            for l in range(0, 7):
                y_den_temp[:, :, l] = gaussian_filter(y_den_temp[:, :, l], self.sigma)
                maxi = np.amax(y_den_temp[:, :, l])
                if maxi != 0:
                    y_den_temp[:, :, l] = y_den_temp[:, :, l] / maxi

            if self.aug_proba:
                con_temp = np.concatenate((img_temp, y_bin_temp, y_den_temp), 2)

                # apply augmentation on all channels
                con_temp = seq_con(image=con_temp)
                # extract img from all channels
                img_temp = con_temp[:, :, :3]
                # apply augmentation for img
                img_temp = seq_img(image=img_temp.astype('uint8'))
                y_bin_temp = con_temp[:, :, 3:10]
                y_den_temp = con_temp[:, :, 10:]
            
            x[i, ] = img_temp
            y_bin[i, ] = y_bin_temp
            y_den[i, ] = y_den_temp

        # NHWC to NCHW
        x = tf.transpose(x,[0, 3, 1, 2])
        y_bin = tf.transpose(y_bin,[0, 3, 1, 2])
        y_den = tf.transpose(y_den,[0, 3, 1, 2])
        return x, [y_bin, y_den]
