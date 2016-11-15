from __future__ import division
import datetime, calendar
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import numpy as np
import copy

# Converts data to CRF++ readable format
def convertToCRFFormat(data_set, label_set, filename):
	outputfile = open(filename,'w')
	seq_no = 0
	for feature in data_set:
		print feature, label_set[seq_no]
		outputfile.write(str(seq_no)  + '\t' + feature[0] + '\t' + feature[1] + '\t' +feature[2] + '\t' + 
			feature[3] + '\t'  + feature[5] + '\t' +feature[6] + '\t' +  feature[7] + '\t' + label_set[seq_no])
		outputfile.write('\n')
		seq_no = seq_no + 1
	outputfile.close()

# Discretizes start time and end time to: 
#	early morning (0-5), morning (5-10), afternoon (10-15), evening (15-20), late evening (20-24)
def getTime(date):
	time = None
	if date.time() <= datetime.time(5) and date.time() >= datetime.time(0):
		time = 'early_morning'
	elif date.time() <= datetime.time(10) and date.time() > datetime.time(5):
		time = 'morning'
	elif date.time() <= datetime.time(15) and date.time() > datetime.time(10):
		time = 'afternoon'
	elif date.time() <= datetime.time(20) and date.time() > datetime.time(15):
		time = 'evening'
	elif date.time() <= datetime.time(23, 59) and date.time() > datetime.time(20):
		time = 'late_evening'
	return time

# Parses sensor data (observations)
def parseInputData(filename):
	data = []
	inputfile = open(filename)
	line = inputfile.readline() # headers
	line = inputfile.readline() # separator

	line = inputfile.readline().split()  # first line
	while line:
		start_date = datetime.datetime.strptime(line[0] + ' ' + line[1], "%Y-%m-%d %H:%M:%S")
		end_date = datetime.datetime.strptime(line[2] + ' ' + line[3], "%Y-%m-%d %H:%M:%S")

		# Get weekday from date
		start_day = calendar.day_name[start_date.weekday()]
		end_day =calendar.day_name[end_date.weekday()]
		# Get discretized time
		start_time = getTime(start_date)
		end_time = getTime(end_date)

		duration = (end_date - start_date).total_seconds() # in seconds	
		location = line[4]
		sensor = line[5]
		place = line[6]
		
		data.append([start_date, end_date, start_day, end_day, start_time, end_time, duration, location, sensor, place])
		line = inputfile.readline().split()

	inputfile.close()
	return data

# Parses class labels
def parseLabelData(filename, input_data):
	output = []
	temp = []
	inputfile = open(filename)
	line = inputfile.readline().strip()
	line = inputfile.readline().strip().split()

	line = inputfile.readline().split()
	while line:
		start_date = datetime.datetime.strptime(line[0] + ' ' + line[1], "%Y-%m-%d %H:%M:%S")
		end_date = datetime.datetime.strptime(line[2] + ' ' + line[3], "%Y-%m-%d %H:%M:%S")
		temp.append([start_date, end_date, line[4]])
		line = inputfile.readline().split()

	iterr,	index = 0, 0
	while iterr < len(input_data):
		if index == len(temp) - 1:
			output.append(temp[-1][2])
			iterr = iterr + 1
		elif input_data[iterr][0] >= temp[index][0] and input_data[iterr][1] <= temp[index][1]:
			output.append(temp[index][2])
			iterr = iterr + 1
		elif input_data[iterr][0] < temp[index][0]:
			output.append('Idle')
			iterr = iterr + 1
		else:
			index = index + 1
	return output, temp

def partitionData(input_data, output_data):
	train = input_data[:int(len(input_data)*0.7)]
	test = input_data[int(len(input_data)*0.7) + 1:]
	train_label = output_data[:int(len(input_data)*0.7)]
	test_label = output_data[int(len(input_data)*0.7) + 1:]
	return train, test, train_label, test_label

def discretizeData(input_data, output_data):
	#Transforming data into numeric values
	trans = preprocessing.LabelEncoder()
	output_data = trans.fit_transform(output_data)
	
	start_day, end_day, start_time, end_time, duration, location, sensor, place = [], [], [], [], [], [], [], []
	for inp in input_data:
		start_day.append(inp[0])
		end_day.append(inp[1])
		start_time.append(inp[2])
		end_time.append(inp[3])
		duration.append(inp[4])
		location.append(inp[5])
		sensor.append(inp[6])
		place.append(inp[7])
	start_day = trans.fit_transform(start_day)
	end_day = trans.fit_transform(end_day)
	start_time = trans.fit_transform(start_time)
	end_time = trans.fit_transform(end_time)
	location = trans.fit_transform(location)
	sensor = trans.fit_transform(sensor)
	place = trans.fit_transform(place)

	for i in range(len(input_data)):
		input_data[i] = [start_day[i], end_day[i], start_time[i], end_time[i], duration[i], location[i], sensor[i], place[i]]
	return input_data, output_data

# Gaussian Naive Bayes
def GaussianNaiveBayes(train, test, train_label, test_label):
	gnb = GaussianNB()
	y_pred = gnb.fit(train, train_label).predict(test)
	score = 0
	for i in range(len(y_pred) - 1):
		if test_label[i] == y_pred[i]:
			score = score + 1
	return score/len(y_pred)
	
	# Support Vector Machines (SVM)
def SupportVectorMachines(train, test, train_label, test_label):
	clf = svm.SVC()
	clf.fit(train, train_label) 
	y_pred = clf.predict(test)
	score = 0
	for i in range(len(y_pred) - 1):
		if test_label[i] == y_pred[i]:
			score = score + 1
	return score/len(y_pred)

def main():
	input_file = 'UCI-ADL-Binary-Dataset/OrdonezB_Sensors.txt'
	input_data =  parseInputData(input_file)

	label_file = 'UCI-ADL-Binary-Dataset/OrdonezB_ADLs.txt'
	output_data, temp = parseLabelData(label_file, input_data)

	for i in range(len(input_data)):
		input_data[i] = input_data[i][2:]

	# Convert training and testing set to CRF++ readable format
	train, test, train_label, test_label = partitionData(input_data, output_data)
	convertToCRFFormat(train, train_label, 'CRF++-0.58/training.txt')
	convertToCRFFormat(test, test_label, 'CRF++-0.58/testing.txt')

	# Other machine learning methods
	new_input_data, new_output_data = discretizeData(input_data, output_data)
	train, test, train_label, test_label = partitionData(new_input_data, new_output_data)
	print "Gaussian Naive Bayes: ", GaussianNaiveBayes(train, test, train_label, test_label)
	print "Support Vector Machines: ", SupportVectorMachines(train, test, train_label, test_label)


if __name__ == '__main__':
    main()