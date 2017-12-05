from PIL import Image
from std import STD_RESOLUTION

class Car:

    def __init__(self, img_path, label):
        self.__img_path = img_path
        self.__img = Image.open(img_path)
        self.__img_arr = self.__toRgbArray()
        self.__label = label


    def __toRgbArray(self):
        arr = []
        raw_data = self.__img.getdata()
        for i in range(STD_RESOLUTION[1]):
            line = []
            for j in range(STD_RESOLUTION[0]):
                line.append(raw_data[i*STD_RESOLUTION[0] + j])

            arr.append(line)

        return arr


    def getImageAsArray(self):
        return self.__img_arr


    
