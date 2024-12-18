import matplotlib.pyplot as plt
import numpy as np
import cv2

def main()->None:
    path = r"E:\GaoXinhe\Research\Data\Im_Processing\dataset_new\val\images_viz\case_00006_46.png"
    im = image_load(path)
    im_eq = equalization(im)
    binary_image = binary_split(im)
    mor_process(binary_image)
    return

def equalization(im):
    equalized_image = cv2.equalizeHist(im)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    image_show(blurred_image,'equalization')
    return blurred_image

def binary_split(im):
    _, binary_image = cv2.threshold(im, 210, 255, cv2.THRESH_BINARY)
    image_show(binary_image,'binary image')
    return binary_image

def edge_detection(im):
    edges = cv2.Canny(binary_image, 50, 150)
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
    image1 = mor_open(im,kernel,1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image2 = mor_open(im,kernel2,2)
    #closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    image_show(image1,'morph_image')
    image_show(image2,'itr = 2')
    return

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