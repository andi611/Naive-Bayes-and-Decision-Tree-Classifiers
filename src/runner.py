# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ runner.py ]
#   Synopsis     [ main program that runs the 'Naive Bayes' and 'Decision Tree' training / testing ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import csv
import argparse
import numpy as np
from data_loader import data_loader
from classifiers import naive_bayes_runner
from classifiers import decision_tree_runner


##################
# CONFIGURATIONS #
##################
def get_config():
	parser = argparse.ArgumentParser(description='descrip_msg')

	classifier = parser.add_argument_group('classifier')
	classifier.add_argument('--classifier', type=str, default='', help='classifier to be specified by user')
	classifier.add_argument('--naive_bayes', action='store_true', help='enable Naive Bayes classification mode')
	classifier.add_argument('--decision_tree', action='store_true', help='enable Decision Tree classification mode')

	mode_args = parser.add_argument_group('mode')
	mode_args.add_argument('--search_opt', action='store_true', help='search for optimal parameters for classifiers')
	mode_args.add_argument('--run_all', action='store_true', help='run all distribution assumption for the Naive Bayes classifier')
	mode_args.add_argument('--visualize_tree', action='store_true', help='plot and visualize the Decision Tree classifier')

	data_args = parser.add_argument_group('data')
	data_args.add_argument('--data_news', action='store_true', help='Training and testing on the News dataset')
	data_args.add_argument('--data_mushroom', action='store_true', help='Training and testing on the Mushroom dataset')
	data_args.add_argument('--data_income', action='store_true', help='Training and testing on the Income dataset')

	path_args = parser.add_argument_group('train_path')
	path_args.add_argument('--train_path', type=str, default='', help='training path to be specified by user')
	path_args.add_argument('--train_path_news', type=str, default='../data/news/news_train.csv', help='path to the News training dataset')
	path_args.add_argument('--train_path_mushroom', type=str, default='../data/mushroom/mushroom_train.csv', help='path to the Mushroom training dataset')
	path_args.add_argument('--train_path_income', type=str, default='../data/income/income_train.csv', help='path to the Income training dataset')

	path_args = parser.add_argument_group('test_path')
	path_args.add_argument('--test_path', type=str, default='', help='testing path to be specified by user')
	path_args.add_argument('--test_path_news', type=str, default='../data/news/news_test.csv', help='path to the News testing dataset')
	path_args.add_argument('--test_path_mushroom', type=str, default='../data/mushroom/mushroom_test.csv', help='path to the Mushroom testing dataset')
	path_args.add_argument('--test_path_income', type=str, default='../data/income/income_test.csv', help='path to the Income testing dataset')
	
	path_args = parser.add_argument_group('output_path')
	path_args.add_argument('--output_path', type=str, default='../result/output.csv', help='path to save model prediction')

	args = parser.parse_args()
	args = error_handling(args)
	return args


##################
# ERROR HANDLING #
##################
def error_handling(args):
	if args.classifier != '':
		args.naive_bayes = True if args.classifier == 'N' else False
		args.decision_tree = True if args.classifier == 'D' else False
	if args.naive_bayes and args.decision_tree == True:
		raise AssertionError('Please choose one classifier at once, or specify the correct classifier!')
	if args.search_opt and args.run_all and args.visualize_tree == True:
		raise AssertionError('Please choose one mode at a time!')
	if args.data_news and args.data_mushroom and args.income == True:
		raise AssertionError('Please choose one and at least one dataset at a time!')
	if args.train_path != '' and args.test_path != '':
		if not os.path.isfile(args.train_path) or not os.path.isfile(args.test_path): 
			raise AssertionError('The given file path is invalid!')
		if args.data_news: 
			args.train_path_news = args.train_path
			args.test_path_news = args.test_path
		elif args.data_mushroom: 
			args.train_path_mushroom = args.train_path
			args.test_path_mushroom = args.test_path
		elif args.data_income: 
			args.train_path_income = args.train_path
			args.test_path_income = args.test_path
		else: 
			raise AssertionError('Must choose a dataset!')
	return args


#################
# OUTPUT WRITER #
#################
def output_writer(path, result):
	with open(path, 'w') as f:
		file = csv.writer(f, delimiter=',', quotechar='\r')
		for item in result:
			file.writerow([int(item)])
	print('Results have been successfully saved to: %s' % (path))
	return True


########
# MAIN #
########
"""
    main function
"""
def main():
	
	args = get_config()
	loader = data_loader(args)

	#---fetch data---#
	if args.data_news:
		train_x, train_y, test_x, test_y = loader.fetch_news()
		MODEL = 'NEWS'
	elif args.data_mushroom:
		train_x, train_y, test_x, test_y = loader.fetch_mushroom()
		MODEL = 'MUSHROOM'
	elif args.data_income:
		train_x, train_y, test_x, test_y = loader.fetch_income() # -> test_y == None
		MODEL = 'INCOME'

	###############
	# NAIVE BAYES #
	###############
	if args.naive_bayes:
		#---construct model---#
		naive_bayes = naive_bayes_runner(MODEL, train_x, train_y, test_x, test_y)

		#---modes---#
		if args.search_opt:
			naive_bayes.search_alpha()
		elif args.run_all:
			naive_bayes.run_best_all()
		else:
			pred_y = naive_bayes.run_best()
			output_writer(args.output_path, pred_y)

	#################
	# DECISION TREE #
	#################
	if args.decision_tree:
		#---construct model---#
		decision_tree = decision_tree_runner(MODEL, train_x, train_y, test_x, test_y)

		#---modes---#
		if args.search_opt:
			decision_tree.search_max_depth()
		elif args.visualize_tree:
			decision_tree.visualize()
		else:
			pred_y = decision_tree.run_best()
			output_writer(args.output_path, pred_y)


if __name__ == '__main__':
	main()

