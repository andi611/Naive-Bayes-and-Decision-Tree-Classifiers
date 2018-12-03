# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ classifiers.py ]
#   Synopsis     [ 'Naive Bayes' and 'Decision Tree' training, testing, and tunning functions ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import tree


############
# CONSTANT #
############
N_FOLD = 10
DEPTHS = np.arange(1, 64)
ALPHAS = np.arange(0.001, 1.0, 0.001)
ALPHAS_MUSHROOM = np.arange(0.0001, 1.0, 0.0001)
BEST_DISTRIBUTION = 'Multinominal'



###############
# NAIVE BAYES #
###############
class naive_bayes_runner(object):

	def __init__(self, MODEL, train_x, train_y, test_x, test_y):
		
		#---data---#
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y

		#---model---#
		self.cross_validate = False
		self.MODEL = MODEL

		if self.MODEL == 'NEWS':
			self.models = {	'Guassian' : GaussianNB(),
					  	 	'Multinominal' : MultinomialNB(alpha=0.065),
							'Complement' : ComplementNB(alpha=0.136),
						 	'Bernoulli' : BernoulliNB(alpha=0.002) }
		if self.MODEL == 'MUSHROOM':
			ALPHAS = ALPHAS_MUSHROOM
			self.models = {	'Guassian' : GaussianNB(),
					  	 	'Multinominal' : MultinomialNB(alpha=0.0001),
							'Complement' : ComplementNB(alpha=0.0001),
						 	'Bernoulli' : BernoulliNB(alpha=0.0001) }
		if self.MODEL == 'INCOME':
			self.cross_validate = True
			self.models = {	'Guassian' : GaussianNB(),
					  	 	'Multinominal' : MultinomialNB(alpha=0.959),
							'Complement' : ComplementNB(alpha=0.16),
						 	'Bernoulli' : BernoulliNB(alpha=0.001) }


	def _fit_and_evaluate(self, model):
		model_fit = model.fit(self.train_x, self.train_y)
		pred_y = model_fit.predict(self.test_x)
		acc = metrics.accuracy_score(self.test_y, pred_y)
		return acc, pred_y
	

	def search_alpha(self):
		try:
			from tqdm import tqdm
		except:
			raise ImportError('Failed to import tqdm, use the following command to install: pip3 install tqdm')
		for distribution, model in self.models.items():
			best_acc = 0.0
			best_alpha = 0.001
			if distribution != 'Guassian': 
				print('>> [Naive Bayes Runner] Searching for best alpha value, distribution:', distribution)
				for alpha in tqdm(ALPHAS):
					model.set_params(alpha=alpha)
					if self.cross_validate: 
						scores = cross_val_score(model, self.train_x, self.train_y, cv=N_FOLD, scoring='accuracy')
						acc = scores.mean()
					else:
						acc, _ = self._fit_and_evaluate(model)
					if acc > best_acc:
						best_acc = acc
						best_alpha = alpha
				print('>> [Naive Bayes Runner] '+ distribution + ' - Best Alpha Value:', best_alpha)


	def run_best_all(self):
		for distribution, model in self.models.items():
			if self.cross_validate: 
				scores = cross_val_score(model, self.train_x, self.train_y, cv=N_FOLD, scoring='accuracy')
				acc = scores.mean()
			else:
				acc, _ = self._fit_and_evaluate(model)
			print('>> [Naive Bayes Runner] '+ distribution + ' - Accuracy:', acc)


	def run_best(self):
		if self.cross_validate: 
			scores = cross_val_score(self.models[BEST_DISTRIBUTION], self.train_x, self.train_y, cv=N_FOLD, scoring='accuracy')
			acc = scores.mean()
			model_fit = self.models[BEST_DISTRIBUTION].fit(self.train_x, self.train_y)
			pred_y = model_fit.predict(self.test_x)
		else:
			acc, pred_y = self._fit_and_evaluate(self.models[BEST_DISTRIBUTION])
		print('>> [Naive Bayes Runner] '+ BEST_DISTRIBUTION + ' - Accuracy:', acc)
		return pred_y


#################
# DECISION TREE #
#################
class decision_tree_runner(object):
	
	def __init__(self, MODEL, train_x, train_y, test_x, test_y):
		
		#---data---#
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		self.test_y = test_y

		#---model---#
		self.cross_validate = False
		self.MODEL = MODEL

		if self.MODEL == 'NEWS':
			self.model = tree.DecisionTreeClassifier(criterion='gini', 
													 splitter='random', 
													 max_depth=47,
													 random_state=1337)
		elif self.MODEL == 'MUSHROOM':
			self.model = tree.DecisionTreeClassifier(criterion='gini', 
													 splitter='random', 
													 max_depth=7,
													 random_state=1337)
		elif self.MODEL == 'INCOME':
			self.cross_validate = True
			self.model = tree.DecisionTreeClassifier(criterion='entropy',  
													 min_impurity_decrease=2e-4,
													 max_depth=15,
													 random_state=1337)


	def _fit_and_evaluate(self):
		model_fit = self.model.fit(self.train_x, self.train_y)
		pred_y = model_fit.predict(self.test_x)
		acc = metrics.accuracy_score(self.test_y, pred_y)
		return acc, pred_y


	def search_max_depth(self):
		try:
			from tqdm import tqdm
		except:
			raise ImportError('Failed to import tqdm, use the following command to install: $ pip3 install tqdm')
		best_acc = 0.0
		best_depth = 1
		
		print('>> [Naive Bayes Runner] Searching for best max depth value...')
		for depth in tqdm(DEPTHS):
			self.model.set_params(max_depth=depth)
			if self.cross_validate: 
				scores = cross_val_score(self.model, self.train_x, self.train_y, cv=N_FOLD, scoring='accuracy')
				acc = scores.mean()
			else:
				acc, _ = self._fit_and_evaluate()
			if acc > best_acc:
				best_acc = acc
				best_depth = depth
		print('>> [Decision Tree Runner] - Best Dpeth Value:', best_depth)


	def visualize(self):
		try:
			import graphviz
		except:
			raise ImportError('Failed to import graphviz, use the following command to install: $ pip3 install graphviz, and $ sudo apt-get install graphviz')
		model_fit = self.model.fit(self.train_x, self.train_y)
		dot_data = tree.export_graphviz(model_fit, out_file=None, 
										filled=True, rounded=True,  
										special_characters=True)  
		graph = graphviz.Source(dot_data)
		graph.format = 'png'
		graph.render('../image/TREE_' + self.MODEL)
		print('>> [Decision Tree Runner] - Tree visualization complete.')


	def run_best(self):
		if self.cross_validate: 
			scores = cross_val_score(self.model, self.train_x, self.train_y, cv=N_FOLD, scoring='accuracy')
			acc = scores.mean()
			model_fit = self.model.fit(self.train_x, self.train_y)
			pred_y = model_fit.predict(self.test_x)
		else:		
			acc, pred_y = self._fit_and_evaluate()
		print('>> [Decision Tree Runner] - Accuracy:', acc)
		return pred_y

