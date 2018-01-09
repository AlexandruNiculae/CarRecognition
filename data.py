import os
import scipy.io
import numpy as np
from pathlib import Path
from car import Car


class Data:

	def __init__(self):
		self.__save_train_folder = 'data\\train_data'
		self.__save_test_folder = 'data\\test_data'
		self.__mat_file = 'data\\cars_annos.mat'
		self.__data_label = 'annotations'
		self.__classes_label = 'class_names'

		self.__max_count = 16185
		self.__image_count = 1000


	def getData(self):
		relocated = self.checkIfRelocated()
		if relocated:
			return self.filterData()
		else:
			return self.readFromMatFile()


	def checkIfRelocated(self):
		return Path(self.__save_train_folder).exists() and Path(self.__save_test_folder).exists()

	def filterData(self):
		cars = self.readFromMatFile()
		for car in cars:
			car_relocated = car.checkIfRelocated()
			if car_relocated:
				car.setPath(car.relocationPath())
		return cars


	def readFromMatFile(self):
		idk = scipy.io.loadmat(self.__mat_file)
		all_cars = []
		cars = idk[self.__data_label].tolist()[0]
		classes = idk[self.__classes_label].tolist()[0]

		i = 0
		for line in cars:
			if i == self.__image_count:
				break
			path = "data/" + line[0][0]
			x1 = line[1][0][0]
			y1 = line[2][0][0]
			x2 = line[3][0][0]
			y2 = line[4][0][0]
			label = line[5][0][0] - 1
			class_name = classes[label][0]
			test = False
			if line[6][0][0] == 1:
				test = True

			car = Car(path,x1,y1,x2,y2,label,class_name,test)
			all_cars.append(car)
			i+=1

		return all_cars

	def getRawData(self):
		return self.__cars


	def relocate(self):
		i = 0
		for car in self.__cars:
			car.relocate()
			i+=1
			print("Images relocated: ",i," out of ",self.__image_count)

	def getKerasDataset(self):
		cars_train = []
		labels_train = []

		cars_test = []
		labels_test = []

		i = 0
		for car in self.getData():
			if not car.isTest():
				cars_train.append(car.asArray())
				labels_train.append(car.getLabel())
			else:
				cars_test.append(car.asArray())
				labels_test.append(car.getLabel())
			i+=1
			print("Images loaded: ",i," out of ",self.__image_count)

		train = (np.array(cars_train),np.array(labels_train))
		test = (np.array(cars_test),np.array(labels_test))

		return train,test




	def getTensorflowDataset(self):
		pass
