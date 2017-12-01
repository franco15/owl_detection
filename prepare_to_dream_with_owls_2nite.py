import sys
import os
import cv2
import csv

def resume(args):
	if not os.path.isdir(args[1]):
		print ('not a directory')
		return
	with open(('csv_files/' + args[2] + '.csv'), 'a') as myfile:
		wr = csv.writer(myfile, delimiter='|')
		# if len(sys.argv) > 2:
		# 	print ('continue from: ' + args[2])
		i = int(args[3])
		for i in range(int(args[3]), len(os.listdir(args[1]))):
			path = args[1] + os.listdir(args[1])[i]
			print (path)
			if not path.endswith(('.JPG', '.jpg', '.png' '.PNG')):
				print (path + ': not an image')
				continue
			img = cv2.imread(path)
			t_img = cv2.resize(img, (900, 600))
			cv2.imshow('owls', t_img)
			while True:
				key = cv2.waitKey(0)
				if key is 27:
					break
				elif key is 48:
					break
				elif key is 49:
					break
			if key is 27: #esc key
				print (i)
				break
			elif key is 48: # key '0'
				wr.writerow((path, '0'))
			elif key is 49: #key '1'
				wr.writerow((path, '1'))
			cv2.destroyWindow('owls')


def ready_question_mark(args):
	if not os.path.isdir(args[1]):
		print ('not a directory')
		return
	with open(('csv_files/' + args[2] + '.csv'), 'w') as myfile:
		wr = csv.writer(myfile, delimiter='|')
		# if len(sys.argv) > 2:
		# 	print ('continue from: ' + args[2])
		i = int(args[3])
		for i in range(int(args[3]), len(os.listdir(args[1]))):
			path = args[1] + os.listdir(args[1])[i]
			print (path)
			if not path.endswith(('.JPG', '.jpg', '.png' '.PNG')):
				print (path + ': not an image')
				continue
			img = cv2.imread(path)
			t_img = cv2.resize(img, (900, 600))
			cv2.imshow('owls', t_img)
			while True:
				key = cv2.waitKey(0)
				if key is 27:
					break
				elif key is 48:
					break
				elif key is 49:
					break
			if key is 27: #esc key
				print (i)
				break
			elif key is 48: # key '0'
				wr.writerow((path, '0'))
			elif key is 49: #key '1'
				wr.writerow((path, '1'))
			cv2.destroyWindow('owls')

def nowls(args):
	if not os.path.isdir(args[1]):
		print ('not a directory')
		return
	with open(('csv_files/' + args[2] + '.csv'), 'w') as myfile:
		wr = csv.writer(myfile, delimiter='|')
		for i in range(0, len(os.listdir(args[1]))):
			path = args[1] + os.listdir(args[1])[i]
			if not path.endswith(('.JPG', '.jpg', '.PNG', '.png')):
				print(path + ': not an image')
				continue
			wr.writerow((path, '0'))

if __name__ == "__main__":
	if len(sys.argv) is 4:
		if os.path.exists('csv_files/' + sys.argv[2] + '.csv'):
			resume(sys.argv)
		else:
			ready_question_mark(sys.argv)
	elif len(sys.argv) is 3:
		nowls(sys.argv)
	else:
		print ('not enough arguments')
		print ('usage: prepare_to_dream_with_owls_2nite.py [src directory] [csv file name] [where to start*]')
		print ('* where to start: ex: 0 to start on first file | 9 to start on 10th file')
