import matplotlib.pyplot as plt
import cv2
import numpy as np
import os.path as osp

def draw_heat_map(in_list, im_path=None, savename=None):

    height = 2
    width = 4
    for j in range(len(in_list)):
        x = in_list[j]
        for k in range(x.size(0)):
            path = im_path[k].replace('\\', '/').split('/')[-1].split('.')[0]
            save_path = str(savename + '/' + str(path) + '_' + str(j))
            # if osp.exists(save_path):
            #     continue
            fig = plt.figure(figsize=(16, 16))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
            for i in range(height * width):
                plt.subplot(height, width, i + 1)
                plt.axis('off')
                img = x[k, i, :, :].cpu().numpy()
                pmin = np.min(img)
                pmax = np.max(img)
                img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
                img = img.astype(np.uint8)  # 转成unit8
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
                img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
                plt.imshow(img)
                # print("{}/{}".format(i, width * height))
            fig.savefig(save_path, dpi=100)
            fig.clf()
            plt.close()
            # print(save_path, "has been saved!!")
