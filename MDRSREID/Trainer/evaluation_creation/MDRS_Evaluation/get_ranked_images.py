import torch
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import time
import warnings
warnings.filterwarnings("ignore", module="matplotlib")


def get_ranked_images(feat_dict, cfg):
    threshold = 0.85
    query_list_img = feat_dict['query_list_img']
    gallery_list_img = feat_dict['gallery_list_img']

    query_feat = torch.FloatTensor(feat_dict['query_feat']).cuda()
    query_label = feat_dict['query_label']
    gallery_feat = torch.FloatTensor(feat_dict['gallery_feat']).cuda()
    gallery_label = feat_dict['gallery_label']
    for i in range(len(query_list_img)):
        save_reid_path = 'D:/weights_results/HJL-ReID/market_3368_match/' + str(i) + '.png'
        if os.path.exists(save_reid_path):
            print(save_reid_path + "has been existed.")
            continue
        # number for result images
        images_numbers = 10
        # search for better results
        # index, images_meet_threshold_numbers, score = MyUtil.sort_img_by_id(query_feature[i], query_label[i], query_cam[i], gallery_feature,
        # gallery_label, gallery_cam, args.threshold)
        index, images_meet_threshold_numbers, score = sort_img_by_name(qf=query_feat[i],
                                                                       gf=gallery_feat,
                                                                       threshold=threshold)
        # Visualize the rank result

        query_path = query_list_img[i]
        query_label_i = query_label[i]
        print('------------------------------------------------------------------------------------------------')
        print("query path is:{}".format(query_path))
        print('\n')
        # print('Top '+ str(flag + 1)+ ' images are as follow:')

        try:  # Visualize Ranking Result
            # Graphical User Interface is needed
            fig = plt.figure(figsize=(16, 4))
            # if images_meet_threshold_numbers < images_numbers:
            #     images_numbers = images_meet_threshold_numbers
            #     if images_meet_threshold_numbers > 0:
            #         print('Top ' + str(images_numbers) + ' images are as follow( accuracy more than {}% ):'.format(
            #             threshold * 100))
            #     else:
            #         print('No image that satisfies {}%!!'.format(threshold * 100))
            ax = plt.subplot(1, images_numbers + 1, 1)
            ax.axis('off')
            imshow(query_path, 'query')
            # for ii in range(flag):

            for ii in range(images_numbers):
                ax = plt.subplot(1, images_numbers + 1, ii + 2)
                # ax = plt.subplot(1, 11, ii + 2)
                ax.axis('off')
                img_path = gallery_list_img[index[ii]]
                label = gallery_label[index[ii]]
                imshow(img_path)
                if label == query_label_i:
                    ax.set_title('{:.1%}'.format(score[index[ii]]), color='green')
                else:
                    ax.set_title('{:.1%}'.format(score[index[ii]]), color='red')
                print('Top ' + str(ii + 1) + ' image : ' + img_path)
            print('------------------------------------------------------------------------------------------------')
            print('\n')
        except RuntimeError:
            print('If you want to see the visualization of the ranking result, graphical user interface is needed.')


        fig.savefig(save_reid_path)
        fig.clf()
        print(save_reid_path + "has been saved.")
    print('Save finish!!')


def sort_img_by_name(qf, gf, threshold):
    query = qf.view(-1, 1)  # [2048]->[2048,1]
    # print(query.shape)
    score = torch.mm(gf, query)  # [114,2048]x[2048,1]=[114,1]
    score = score.squeeze(1).cpu()  # 去掉第2维，[114]
    score = score.numpy()  # (114,)
    # predict index
    index = np.argsort(score)  # from small to large,返回数组值从小到大的索引值
    index = index[::-1]  # from large to small

    images_numbers = 0
    for i in range(score.size):
        if score[index[i]] < threshold:
            break
        else:
            images_numbers += 1
    return index, images_numbers, score


def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated


def save_feat_dict_mat(feat_dict):
    feat_dict_path = 'D:/weights_results' + '/pytorch_result.mat'
    scipy.io.savemat(feat_dict_path, feat_dict)