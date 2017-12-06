from PIL import Image
from std import STD_RESOLUTION,STD_WIDTH,STD_HEIGHT

class Car:

    def __init__(self, img_path, label):
        self.__img_path = img_path
        self.__img = Image.open(img_path)
        self.__label = label


    def getImageAsArray(self):
        return self.__img.getdata()


    def getResolution(self):
        return STD_RESOLUTION
