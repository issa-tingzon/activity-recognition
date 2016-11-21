from __future__ import division
import datetime, calendar
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import preprocessing
import numpy as np
import copy
import pandas 
import os 
import subprocess
import evaluate

# Converts data to CRF++ readable format
def convertToCRFFormat(data_set, label_set, filename):
	outputfile = open(filename,'w')
	seq_no = 0
	start = data_set[0][1]
	for feature in data_set:
		if feature[1] != start:
			outputfile.write('\n')
			start = feature[1]
		outputfile.write(str(seq_no) + '\t' + str(feature[1]) + '\t' + str(feature[2]) + '\t' + 
			str(feature[3]) + '\t'  + str(feature[4]) + '\t'+ str(feature[5]) + '\t'+ str(feature[6]) + '\t'+ str(feature[7]) + 
			'\t'+ str(feature[8]) + '\t'+ str(feature[9]) + '\t'+ str(feature[10]) + '\t'+ str(feature[11]) + '\t'+ str(feature[12]) + 
			'\t'+ str(feature[13]) + '\t'+ str(feature[14]) +  '\t' + label_set[seq_no])
		outputfile.write('\n')
		seq_no = seq_no + 1
	outputfile.close()

# Discretizes time: early morning (0-5), morning (5-10), afternoon (10-15), evening (15-20), late evening (20-24)
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
	elif date.time() <= datetime.time(23, 59, 59) and date.time() > datetime.time(20):
		time = 'late_evening'
	return time

# Sets the feature values for each data entry
def setFeatures(data, sensor_list):
	input_data = []

	for inp in data:
		sensors = inp[3]
		sensor_values = []
		for sensor in sensor_list:
			if sensor in sensors:
				sensor_values.append(1)
			else:
				sensor_values.append(0)
		entry = []
		entry.append(inp[0])
		entry.append(inp[1])
		entry.append(inp[2])
		for value in sensor_values:
			entry.append(value)
		input_data.append(entry)
	return input_data

# Parses sensor data (observations)
def parseInputData(filename):
	data = []
	inputfile = open(filename)
	line = inputfile.readline() # headers
	line = inputfile.readline() # separator
	line = inputfile.readline().split()  # first line
	curr, end = datetime.datetime.strptime(line[0] + ' ' + line[1], "%Y-%m-%d %H:%M:%S"), None

	while line:
		start_date = datetime.datetime.strptime(line[0] + ' ' + line[1], "%Y-%m-%d %H:%M:%S")
		end_date = datetime.datetime.strptime(line[2] + ' ' + line[3], "%Y-%m-%d %H:%M:%S")
		if line[4] == 'Door':
			sensor_type = line[4] + ' ' + line[6]
		else: sensor_type = line[4]
		data.append([start_date, end_date, sensor_type])
		line = inputfile.readline().split()
		if not line:
			end = end_date
	inputfile.close()

	# Segment data into time slices of length 60 sec
	input_data = []
	while curr < end:
		day = calendar.day_name[curr.weekday()]
		time = getTime(curr)
		#sensors = []
		for element in data:
			if 	(curr + datetime.timedelta(seconds=59) >= element[1] and curr  <= element[1]) or \
				(curr <= element[0] and curr + datetime.timedelta(seconds=59) >= element[0]) or \
				(curr  < element[0] and curr + datetime.timedelta(seconds=59) > element[1]):
				sensors = [element[2]] #sensors.append(element[2]) 
		input_data.append([curr, day, time, sensors])
		curr = curr + datetime.timedelta(seconds=60)
	return input_data


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
		elif (input_data[iterr][0] >= temp[index][0] and input_data[iterr][0] <= temp[index][1]) or \
		 	 (input_data[iterr][0] + datetime.timedelta(seconds=59) >= temp[index][0] and input_data[iterr][0] + datetime.timedelta(seconds=59) <= temp[index][1]) or \
		 	 (input_data[iterr][0] < temp[index][0] and input_data[iterr][0] + datetime.timedelta(seconds=59) > temp[index][1]):
			output.append(temp[index][2])
			iterr = iterr + 1
		elif input_data[iterr][0] < temp[index][0]:
			output.append('Idle')
			iterr = iterr + 1
		else: index = index + 1
	return output, temp

#Returns a list that partitions the data by date
def partitionDays(input_data, output_data, no_days):
	start = input_data[0][0].date()
	inp, out = [], []
	for i in range(no_days):
		inp.append([])
		out.append([])
	iterr = 0
	for i in range(len(input_data)):
		if input_data[i][0].date() == start:
			inp[iterr].append(input_data[i])
			out[iterr].append(output_data[i])
		else:
			start = input_data[i][0].date()
			iterr = iterr + 1
			if iterr >= no_days:
				inp[iterr-1].append(input_data[i])
				out[iterr-1].append(output_data[i])
				break
			else:
				inp[iterr].append(input_data[i])
				out[iterr].append(output_data[i])
	return inp, out

# 'leave-one-day-out' cross validation
def partitionData(input_days, output_days, n):
	train, test, train_label, test_label = [], [], [], []
	for day in range(len(input_days)):
		if day == n:
			for i in range(len(input_days[day])):
				test.append(input_days[day][i])
				test_label.append(output_days[day][i])
		else:
			for i in range(len(input_days[day])):
				train.append(input_days[day][i])
				train_label.append(output_days[day][i])		
	return train, test, train_label, test_label

#Transforming continuous data into numeric values
def discretizeData(input_data, output_data):
	trans = preprocessing.LabelEncoder()
	output_data = trans.fit_transform(output_data)
	
	datetime, day, time = [], [], []
	for inp in input_data:
		datetime.append(inp[0])
		day.append(inp[1])
		time.append(inp[2])
	day = trans.fit_transform(day)
	time = trans.fit_transform(time)

	for i in range(len(input_data)):
		sensor_values = input_data[i][3:]
		input_data[i] = [datetime[i], day[i], time[i]]
		for value in sensor_values:
			input_data[i].append(value)
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

#Decision Tree Classifier
def DecisionTrees(train, test, train_label, test_label):
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(train, train_label)
	y_pred = clf.predict(test)
	score = 0
	for i in range(len(y_pred) - 1):
		if test_label[i] == y_pred[i]:
			score = score + 1
	return score/len(y_pred)

# Remove idle/unlabelled states
def removeIdleStates(input_data, output_data):
	inp, out = copy.deepcopy(input_data), copy.deepcopy(output_data)
	new_index = 0
	for index in range(len(output_data)):
		if output_data[index] == 'Idle':
			del inp[new_index]
			del out[new_index]
			new_index = new_index - 1
		new_index = new_index + 1
	return inp, out

def main():
	no_sensors = 12
	choice = 'A'
	if choice == 'A':
		no_days = 14
		sensor_list = ['Shower', 'Basin', 'Cooktop', 'Maindoor', 'Fridge', 'Cabinet', 'Cupboard', 'Toilet', 'Seat', 'Bed', 'Microwave', 'Toaster'] #OrdonezA
		input_file = 'UCI-ADL-Binary-Dataset/OrdonezA_Sensors.txt'
		label_file = 'UCI-ADL-Binary-Dataset/OrdonezA_ADLs.txt'
	elif choice=='B':
		no_days = 21
		sensor_list = ['Shower', 'Basin', 'Door Kitchen', 'Door Living', 'Door Bedroom', 'Maindoor','Fridge', 'Cupboard', 'Toilet', 'Seat', 'Bed', 'Microwave'] #OrdonezB
		input_file = 'UCI-ADL-Binary-Dataset/OrdonezB_Sensors.txt'
		label_file = 'UCI-ADL-Binary-Dataset/OrdonezB_ADLs.txt'

	input_data =  parseInputData(input_file)
	output_data, temp = parseLabelData(label_file, input_data)
	# Print input and label data 
	for i in range(len(input_data)):
		print input_data[i], output_data[i]
	print len(input_data)

	# Sets features
	input_data = setFeatures(input_data, sensor_list)
	# Remove idle states
	input_data, output_data = removeIdleStates(input_data, output_data)
	# Partition days for 'leave-one-day-out' cross validation
	input_days, output_days = partitionDays(input_data, output_data, no_days)

	# Cross Validation
	gb, svm, dt, crf = 0, 0, 0, 0
	for i in range(no_days):
		# Convert training and testing set to CRF++ readable format
		train, test, train_label, test_label = partitionData(input_days, output_days, i)
		convertToCRFFormat(train, train_label, 'CRF++-0.58/training.txt')
		convertToCRFFormat(test, test_label, 'CRF++-0.58/testing.txt')
		
		cwd = os.getcwd() + '\CRF++-0.58'
		subprocess.call([cwd+'\crf_learn.exe',  cwd+'\\template',  cwd+'\\training.txt',  cwd+'\model'])
		res = os.getcwd()+'\\result.txt'
		with open(res, "w+") as output:
			subprocess.call([cwd+'\crf_test.exe', '-m', cwd+'\\model',  cwd+'\\testing.txt'],  stdout=output)
		true_output, predicted_output = evaluate.readResults(res)
		crf = crf + evaluate.evaluate(true_output, predicted_output)

		# Data partitioning for other machine learning methods
		new_input_data, new_output_data = discretizeData(input_data, output_data)
		new_input_days, new_output_days = partitionDays(new_input_data, new_output_data, no_days)
		train, test, train_label, test_label = partitionData(new_input_days, new_output_days, i)
		# Remove the first entry (datetime entry)
		for i in range(len(train)):
			train[i] = train[i][1:]
		for i in range(len(test)):
			test[i] = test[i][1:]
		# Apply other machine learning methods
		gb = gb + GaussianNaiveBayes(train, test, train_label, test_label)
		svm = svm + SupportVectorMachines(train, test, train_label, test_label)
		dt = dt + DecisionTrees(train, test, train_label, test_label)
		print gb, svm, dt, crf

	print "Gaussian Naive Bayes: ", gb/no_days
	print "Support Vector Machines: ", svm/no_days
	print "Decision Tree: ", dt/no_days
	print "Conditional Random Field: ", crf/no_days

if __name__ == '__main__':
    main()