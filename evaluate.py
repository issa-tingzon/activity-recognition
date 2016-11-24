from __future__ import division

def calculateTPTI(true_output, predicted_output):
	TP = dict()
	TI = dict()
	TT = dict()
	size = len(true_output)

	for i in range(size):
		true = true_output[i]
		predicted = predicted_output[i]

		if true not in TP:
			TP[true] = 0
		if predicted not in TP:
			TP[predicted] = 0

		if true == predicted: #True positives
			TP[true] = TP[true] + 1
		elif predicted != true:
			if predicted not in TI:
				TI[predicted] = 1
			else:
				TI[predicted] = TI[predicted] + 1

			if true not in TT:
				TT[true] = 1
			else:
				TT[true] = TT[true] + 1

	for label in TP:
		if label not in TI:
			TI[label] = TP[label]
		else: TI[label] = TP[label] + TI[label]

	for label in TP:
		if label not in TT:
			TT[label] = TP[label]
		else: TT[label] = TP[label] + TT[label]

	return TP, TI, TT

def calculateFmeasure(true_output, predicted_output):
	TP, TI, TT = calculateTPTI(true_output, predicted_output)
	precision = 0
	recall = 0

	for label in TP:
		if TP[label] > 0:
	 		precision = precision + (TP[label]/TI[label])
			recall = recall + (TP[label]/TT[label])
	precision = precision/len(TP)
	recall = recall/len(TP)

	if (precision + recall) > 0:
		Fmeasure = (2*precision*recall)/(precision + recall)
	else: Fmeasure = 0
	return Fmeasure



def evaluateAccuracy(true_output, predicted_output):
	total = 0
	correct = 0
	size = len(true_output)

	#print true_output, predicted_output

	for i in range(size):
		true = true_output[i]
		predicted = predicted_output[i]

		if true == predicted:
			correct = correct + 1
		total = total + 1

	return correct/total

def readResults(filename):
	file = open(filename)
	line = file.readline()

	true_output = []
	predicted_output = []
	true_output_set = []
	
	while line:
		line = line.split()
		if line == []:
			break
		true_output.append(line[-2])
		predicted_output.append(line[-1])
		line = file.readline()	

	file.close()
	return true_output, predicted_output

def main():
	filename = 'results.txt'
	true_output, predicted_output = readResults(filename)
	accuracy = evaluate(true_output, predicted_output)
	print accuracy


if __name__ == '__main__':
    main()