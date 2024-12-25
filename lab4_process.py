import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.measure import regionprops
import numpy as np
import cv2

def main()->None:
    path = r"E:\GaoXinhe\Research\Data\Im_Processing\dataset_new\val\images_viz\case_00188_148.png"
    im = image_load(path)
    im_eq = equalization(im)
    binary_image = binary_split(im_eq)
    mor_image = mor_process(binary_image)
    seperate(mor_image)
    return

def seperate(im):
    labeled_image, num_features = ndimage.label(im)
    plt.imshow(labeled_image, cmap='nipy_spectral')  # 使用不同颜色显示不同连通域
    plt.title("Labeled Connected Components")
    plt.show()
    target_regions = np.zeros_like(im)
    regions = regionprops(labeled_image)
    sorted_regions = sorted(regions, key=lambda region: region.area, reverse=True)
    found_count = 0
    for region in sorted_regions:
        flag = if_target(region.centroid)
        found_count += flag
        target_regions[labeled_image == region.label ] = flag
        print(f"Region {region.label}: Area = {region.area}, Centroid = {region.centroid},target = {flag}")
        if found_count == 2:
            break
    plt.imshow(target_regions, cmap='gray')
    plt.title("Selected Connected Components")
    plt.show()
    return

def if_target(centroid:tuple)-> bool :
    lbndry_1 = 100
    rbndry_1 = 200
    lbndry_2 = 300
    rbndry_2 = 400
    ubndry = 250
    dbndry = 350
    y = centroid[0]
    x = centroid[1]
    flag = (x>lbndry_1)&(x<rbndry_1)&(y>ubndry)&(y<dbndry)|(x>lbndry_2)&(x<rbndry_2)&(y>ubndry)&(y<dbndry)
    return flag


def equalization(im):
    equalized_image = cv2.equalizeHist(im)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    image_show(blurred_image,'equalization')
    return blurred_image

def binary_split(im):
    _, binary_image = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY)
    image_show(binary_image,'binary image')
    return binary_image

def edge_detection(im):
    edges = cv2.Canny(im, 50, 150)
    image_show(edges,'edge detection')
    return edges

def mor_open(image,kernel,itr):
    erosion = cv2.erode(image,kernel,iterations = itr)
    dilation = cv2.dilate(erosion,kernel,iterations = itr)
    return dilation

def mor_close(image,kernel,itr):
    dilation = cv2.dilate(image,kernel,iterations = itr)
    erosion = cv2.erode(dilation,kernel,iterations = itr)
    return erosion

def mor_process(im):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = mor_open(im,kernel,1)
    #closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    image_show(image,'morph_image')
    return image

def image_show(im,title='default title')->None:
    plt.imshow(im, cmap='gray') 
    plt.axis('off')
    plt.title(title)
    plt.show()
    return

def image_load(im_path):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    #im = np.array(im, dtype=np.double)
    #print(im.shape)
    #print(dir(im))
    return im


if __name__== '__main__':
    main()
