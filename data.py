from PIL import Image
from std import STD_RESOLUTION

import glob
import os

class DataFilter:

    def __init__(self, main_dir):
        self.__main_dir = main_dir
        self.__data = {}
        self.__images = []

        self.__initData()
        self.__splitData()


    def __initData(self):
        for img_path in glob.glob(self.__main_dir + "\*.jpg"):
            img_temp = img_path.split("\\")
            img_name = img_temp[1].split("_")[0]
            if img_name not in list(self.__data.keys()):
                self.__data[img_name] = []
                self.__data[img_name].append(img_path)
            else:
                self.__data[img_name].append(img_path)

    def __splitData(self):

        for folder in list(self.__data.keys()):
            new_folder = self.__main_dir + "\\" + folder
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

            idx = 0
            for img_path in self.__data[folder]:
                img = Image.open(img_path)
                img = img.resize(STD_RESOLUTION)
                img.save(new_folder+"\\"+folder + "_" + str(idx) +".jpg")
                idx+=1

                self.__images.append((img_path,folder))


    def getLabels(self):
        return set([pair[1] for pair in self.__images])


    def getImages(self):
        return self.__images
