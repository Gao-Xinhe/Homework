import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.measure import regionprops
import numpy as np
import cv2

class ImageLoader:
    def __init__(self,image_path:str):
        self.image_path = image_path
        self.image = self.image_load()
    
    def image_load(self):
        im = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        #im = np.array(im, dtype=np.double)
        #print(im.shape)
        #print(dir(im))
        return im
    
    def image_show(self,im ,title='default title')->None:
        plt.imshow(im, cmap='gray') 
        plt.axis('off')
        plt.title(title)
        plt.show()
        return

class ImageTransformer:
    def __init__(self,image):
        self.image = image
        self.processed_image = None
    
    def equalization(self):
        equalized_image = cv2.equalizeHist(self.image)
        self.processed_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
        # image_show(blurred_image,'equalization')
        return self.processed_image
        
    def binary_split(self):
        _, binary_image = cv2.threshold(self.processed_image, 230, 255, cv2.THRESH_BINARY)
        # image_show(binary_image,'binary image')
        return binary_image
    
    def mor_open(self,image,kernel,itr):
        erosion = cv2.erode(image,kernel,iterations = itr)
        dilation = cv2.dilate(erosion,kernel,iterations = itr)
        return dilation

    def mor_close(self,image,kernel,itr):
        dilation = cv2.dilate(image,kernel,iterations = itr)
        erosion = cv2.erode(dilation,kernel,iterations = itr)
        return erosion

    def mor_process(self,binary_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        image = self.mor_open(binary_image,kernel,1)
        #closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
        #image_show(image,'morph_image')
        return image
    
class ImageAnalyzer:
    def __init__(self,processed_image):
        self.processed_image = processed_image

    def if_target(self,centroid:tuple)-> bool :
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
    
    def seperate(self,image):
        labeled_image, num_features = ndimage.label(image)
        plt.imshow(labeled_image, cmap='nipy_spectral')  # 使用不同颜色显示不同连通域
        plt.title("Labeled Connected Components")
        plt.show()
        target_regions = np.zeros_like(image)
        regions = regionprops(labeled_image)
        sorted_regions = sorted(regions, key=lambda region: region.area, reverse=True)
        found_count = 0
        for region in sorted_regions:
            flag = self.if_target(region.centroid)
            found_count += flag
            target_regions[labeled_image == region.label ] = flag
            print(f"Region {region.label}: Area = {region.area}, Centroid = {region.centroid},target = {flag}")
            if found_count == 2:
                break
        plt.imshow(target_regions, cmap='gray')
        plt.title("Selected Connected Components")
        plt.show()
        return
    
class ImageProcessor:
    def __init__(self,image_path:str):
        self.loader = ImageLoader(image_path)
        self.transformer = None
        self.analyzer = None
    
    def process_image(self):
        #load image
        self.loader.image_show(self.loader.image,'Original Image')
        #process image
        transformer = ImageTransformer(self.loader.image)
        self.transformer = transformer
        processed_image = transformer.equalization()
        binary_image = transformer.binary_split()
        mor_image = transformer.mor_process(binary_image)
        self.transformer.processed_image = mor_image
        #split image
        analyer = ImageAnalyzer(mor_image)
        self.analyzer = analyer
        analyer.seperate(mor_image)


def main()->None:
    path = r"E:\GaoXinhe\Research\Data\Im_Processing\dataset_new\val\images_viz\case_00026_163.png"

    processor = ImageProcessor(path)
    processor.process_image()

    return



if __name__== '__main__':
    main()
