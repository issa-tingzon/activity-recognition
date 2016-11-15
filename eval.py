from __future__ import division


def evaluate(true_output, predicted_output):
	total = 0
	correct = 0
	size = len(true_output)

	print true_output, predicted_output

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