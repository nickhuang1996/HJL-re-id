import numpy as np
import cv2



def create_visual_anno(anno):
    """"""
    assert np.max(anno) <= 7, "only 7 classes are supported, add new color in label2color_dict"
    label2color_dict = {
        0: [0, 0, 0],
        1: [255, 248, 220],  # cornsilk
        2: [100, 149, 237],  # cornflowerblue
        3: [102, 205, 170],  # mediumAquamarine
        4: [205, 133, 63],  # peru
        5: [160, 32, 240],  # purple
        6: [255, 64, 64],  # brown1
        7: [139, 69, 19],  # Chocolate4
    }
    # visualize
    visual_anno = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    for i in range(visual_anno.shape[0]):  # i for h
        for j in range(visual_anno.shape[1]):
            color = label2color_dict[anno[i, j]]
            visual_anno[i, j, 0] = color[0]
            visual_anno[i, j, 1] = color[1]
            visual_anno[i, j, 2] = color[2]

    return visual_anno

if __name__ == '__main__':
    test_img_file = 'F:/datasets/EANet_dataset/' \
                    'market1501/Market-1501-v15.09.15_ps_label/bounding_box_train/' \
                    '0002_c1s1_000451_03.png'
    test_img = cv2.imread(test_img_file, 0)
    print(test_img)
    print(test_img.shape)
    visual_test_img = create_visual_anno(test_img)
    print(visual_test_img)
    cv2.imshow('test_img', visual_test_img)
    cv2.waitKey()
