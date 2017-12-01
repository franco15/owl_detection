import os
import csv
import sys
import glob
import shutil

def move_owls(args):
	l = 1
	for i in range(0, len(os.listdir(args[1]))):
		with open(args[1] + os.listdir(args[1])[i], 'r') as myfile:
			reader = csv.reader(myfile, delimiter='|')
			for row in reader:
				if row[1] is '0':
					print (row[0])
					shutil.copy(row[0], 'not_owls/no_owl_' + str(l) + '.jpg')
				elif row[1] is '1':
					print (row[0])
					shutil.copy(row[0], 'owls/owl_' + str(l) + '.jpg')
				l += 1

if __name__ == "__main__":
	move_owls(sys.argv)
