from PIL import Image
import numpy as np
import os
from pathlib import Path
class Car:

	def __init__ (self,path,x1,y1,x2,y2,label,name,test):
		self.__save_train_folder = 'data\\train_data'
		self.__save_test_folder = 'data\\test_data'


		self.__path = path
		self.__x1 = x1
		self.__y1 = y1
		self.__x2 = x2
		self.__y2 = y2
		self.__label = label
		self.__name = name
		self.__test = test

	def relocate(self):


		img = Image.open(self.__path)
		img = img.resize((300,300))
		img = img.convert('L')

		self.__path = self.relocationPath()
		img.save(self.__path)


	def relocationPath(self):
		base = os.path.basename(self.__path)
		if self.isTest():
			return self.__save_test_folder + "\\" + base
		else:
			return self.__save_train_folder + "\\" + base

	def checkIfRelocated(self):
		return Path(self.relocationPath()).exists()

	def setPath(self,path):
		self.__path = path

	def getPath(self):
		return self.__path

	def getLabel(self):
		return self.__label

	def getClass(self):
		return self.__name

	def isTest(self):
		return self.__test

	def asArray(self):
		img = Image.open(self.__path)
		img = img.resize((300,300))
		img = img.convert('L')
		arr = np.array(img).tolist()
		return arr


	def __str__(self):
		idk = ""
		idk += self.__path
		idk += " -- "
		idk += str(self.__label)
		idk += "-"
		idk += self.__name
		idk += " Test:"
		idk += str(self.__test)
		return idk

	def __repr__(self):
		return str(self)
