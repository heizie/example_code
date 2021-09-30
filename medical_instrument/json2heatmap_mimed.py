# -*- coding: utf-8 -*-
import os, cv2, json
from scipy.ndimage import gaussian_filter
import numpy as np


def check_folder_exist(folder):
    if not os.path.exists(folder):   # check the path
        os.makedirs(folder)          # if not exis, create


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

def jsonlist2map(json_train_dir, json_name, input_h, input_w, scale_h, scale_w, rd, sigma):
    y_bin_temp = np.zeros((input_h, input_w, 7))
    y_den_temp = np.zeros((input_h, input_w, 7))

    json_train_path =  os.path.join(json_train_dir, json_name)
    all_points = read_json(json_train_path, scale_h, scale_w)

    for node in all_points:
        if node[0] == 'left':
            y_bin_temp, y_den_temp = draw_point(rd, 0, node, y_bin_temp, y_den_temp)

        elif node[0] == 'right':
            y_bin_temp, y_den_temp = draw_point(rd, 1, node, y_bin_temp, y_den_temp)

        elif node[0] == 'joint':
            y_bin_temp, y_den_temp = draw_point(rd, 2, node, y_bin_temp, y_den_temp)

        elif node[0] == 'em':
            y_bin_temp, y_den_temp = draw_point(rd, 3, node, y_bin_temp, y_den_temp)

    for j, node in enumerate(all_points):
        # left-joint
        if node[0] == 'left':
            y_bin_temp, y_den_temp = draw_line(rd, 4, all_points, y_bin_temp, y_den_temp, j, skip=2)

        # right-joint
        elif node[0] == 'right':
            y_bin_temp, y_den_temp = draw_line(rd, 5, all_points, y_bin_temp, y_den_temp, j, skip=1)

        # joint-em
        elif node[0] == 'joint':
            y_bin_temp, y_den_temp = draw_line(rd, 6, all_points, y_bin_temp, y_den_temp, j, skip=1)
        
    # gaussian filter for each layer
    for l in range(0, 7):
        y_den_temp[:, :, l] = gaussian_filter(y_den_temp[:, :, l], sigma)
        maxi = np.amax(y_den_temp[:, :, l])
        # print(l, maxi)
        if maxi != 0:
            y_den_temp[:, :, l] = y_den_temp[:, :, l] / maxi

    return y_bin_temp, y_den_temp

def Json2NPY(Json_dir, Heatmap_dir, rd, sigma, input_h, input_w, scale_h, scale_w):
    
    json_train_dir = os.path.join(Json_dir, 'train')
    json_test_dir = os.path.join(Json_dir, 'test')

    heatmap_train_dir = os.path.join(Heatmap_dir, 'train')
    heatmap_test_dir = os.path.join(Heatmap_dir, 'test')

    json_train_list = os.listdir(json_train_dir)
    json_test_list = os.listdir(json_test_dir)

    for json_name in json_train_list:
        y_bin_temp, y_den_temp = jsonlist2map(json_train_dir, json_name, input_h, input_w, scale_h, scale_w, rd, sigma)
        y = np.concatenate((y_bin_temp, y_den_temp), 2)

        heatmap_name = json_name.split('.')[0]
        heatmap_path = os.path.join(heatmap_train_dir, heatmap_name + '.npy') 
        check_folder_exist(heatmap_train_dir)
        np.save(heatmap_path, y)

    print('train OK')
    for json_name in json_test_list:
        y_bin_temp, y_den_temp = jsonlist2map(json_test_dir, json_name, input_h, input_w, scale_h, scale_w, rd, sigma)
        y = np.concatenate((y_bin_temp, y_den_temp), 2)

        heatmap_name = json_name.split('.')[0]
        heatmap_path = os.path.join(heatmap_test_dir, heatmap_name + '.npy') 
        check_folder_exist(heatmap_test_dir)
        np.save(heatmap_path, y)

    print('test OK')


if __name__ == "__main__":
    Heatmap_dir = "/tf/mount/Dataset/mimed/Dataset_15_09_full/Heatmap"
    Json_dir = "/tf/mount/Dataset/mimed/Dataset_15_09_full/Annotations"
    # Img_dir = "/media/lin/文件/Dataset/mimed/Dataset_15_09/JPEGImages"
    img_h = 1030
    img_w = 1320

    input_h = 256
    input_w = 320

    Json2NPY(Json_dir, Heatmap_dir, 10, 20, input_h, input_w, input_h/img_h, input_w/img_w)