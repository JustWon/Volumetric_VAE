import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


nyudv2_color_codes = {
    'empty': [0,0,0],
    'floor': [162, 227, 148], 
    'wall': [183, 205, 235], 
    'cabinet': [35, 131, 188], 
    'bed': [255, 196, 131], 
    'chair': [195, 197, 37], 
    'sofa': [151, 97, 84], 
    'table': [255, 162, 161], 
    'door': [218, 45, 45], 
    'window': [203, 185, 218], 
    'bookshelf': [159, 114, 196], 
    'picture': [203, 166, 158], 
    'counter': [24, 197, 214], 
    'desk': [248, 191, 216], 
    'curtain': [222, 224, 152], 
    'refrigerator': [254, 139, 10], 
    'bathtub':[231, 130, 201], 
    'shower_curtain': [168, 223, 232], 
    'toilet': [48, 170, 51], 
    'sink':[123, 139, 155], 
    'otherfuniture':[93, 93, 173] 
}
    
def rgb2nyudv2(rgb):
    
    color_dist = []
    for class_name, code in nyudv2_color_codes.items():
        color_dist.append(np.linalg.norm(np.asarray(code)-rgb))
    
    class_id = np.argmin(color_dist)
    class_name = list(nyudv2_color_codes.keys())[class_id]
    
    return class_id, class_name


def pcd2voxel(pcd, filter_rad=0.1, dim_size=32):
    [xc, yc, zc] = np.mean(pcd.points,0)
    [xs, ys, zs] = [xc, yc, zc] + np.ones(3)*filter_rad*dim_size/2
    [xe, ye, ze] = [xc, yc, zc] - np.ones(3)*filter_rad*dim_size/2

    voxel_volume = np.zeros((dim_size,dim_size,dim_size))
    color_volume = np.zeros((dim_size,dim_size,dim_size,3))
    for point, color in zip(pcd.points, pcd.colors):
        xi,yi,zi = (np.around(([xs, ys, zs] - point)/filter_rad)).astype(int)
        voxel_volume[xi, yi, zi] = 1
        color_volume[xi, yi, zi]= color
    
    return voxel_volume, color_volume

# should be faster
def colorvol2classidvol(color_volume, onehot=True):
    classid_volume = np.zeros([32,32,32])
    classid_volume = classid_volume.flatten()
    color_volume = color_volume.reshape([-1,3])

    for idx, color_voxel in enumerate(color_volume):
        if (not np.array_equal(color_voxel,[0,0,0])):
            class_id, calss_name = rgb2nyudv2(color_voxel*255)
            classid_volume[idx] = class_id

    classid_volume = classid_volume.reshape([32,32,32])
    
    if (onehot):
        onehot_classid_volume = np.zeros([32,32,32,21])
        onehot_classid_volume = onehot_classid_volume.reshape([-1, 21])
        classid_volume = classid_volume.flatten()
        
        for onehot_classid_voxel, classid_voxel in zip(onehot_classid_volume, classid_volume):
            onehot_classid_voxel[int(classid_voxel)] = 1
            
        onehot_classid_volume = onehot_classid_volume.reshape([32,32,32,21])
        onehot_classid_volume = onehot_classid_volume.transpose([3,0,1,2])
        classid_volume = onehot_classid_volume
    
    return classid_volume

def classidvol2colorvol(classid_volume):
    color_volume = np.zeros([32,32,32,3])
    color_volume = color_volume.reshape([-1,3])
    classid_volume = classid_volume.flatten()
    
    for idx, classid_voxel in enumerate(classid_volume):
        if (classid_voxel != 0):
            color_volume[idx] = list(nyudv2_color_codes.values())[int(classid_voxel)]
    
    color_volume = color_volume.reshape([32,32,32,3])
    
    return color_volume


def viz_vvae_output(output_color_volume, output_dir, divide=True, show=True, save=False):
    voxel_volume = np.zeros([32,32,32])
    voxel_volume = voxel_volume.flatten()
    color_volume = output_color_volume.reshape([-1,3])

    for idx, color_voxel in enumerate(color_volume):
        if (not np.array_equal(color_voxel, [0,0,0])):
            voxel_volume[idx] = 1

    voxel_volume = voxel_volume.reshape([32,32,32])
    color_volume = color_volume.reshape([32,32,32,3])
    
    if (divide):
        color_volume = color_volume/256.
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_volume, facecolors=color_volume)
    
    if (show): plt.show()
    if (save): 
        timestamp = str(datetime.datetime.utcnow())
        file_name = os.path.join(output_dir,timestamp+'.png')
        plt.savefig(file_name)
    
    plt.close()

def convert_correct(input_volume,target_volume):
    flatten_input = input_volume[0].reshape([21,-1])
    flatten_input = flatten_input.cpu()
    flatten_input = flatten_input.numpy().transpose([1,0])

    flatten_target = target_volume[0]
    flatten_target = flatten_target.cpu()
    flatten_target = flatten_target.numpy()
    flatten_target = flatten_target.flatten()

    convert_flag = True
    for i, t in zip(flatten_input, flatten_target):
        if (t != 0):
            if (np.argmax(i) != t):
                convert_flag = False

    return convert_flag