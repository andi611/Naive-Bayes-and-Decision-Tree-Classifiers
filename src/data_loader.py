# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ data_loader.py ]
#   Synopsis     [ Loader that parse the 'News', 'Mushroom', 'Income' dataset for 'Naive Bayes' and 'Decision Tree' algorithms]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.base import TransformerMixin


###############
# DATA LOADER #
###############
class data_loader(object):

	def __init__(self, args):
		
		#---training paths---#
		self.train_path_news = args.train_path_news
		self.train_path_mushroom = args.train_path_mushroom
		self.train_path_income = args.train_path_income
		
		#---testing paths---#
		self.test_path_news = args.test_path_news
		self.test_path_mushroom = args.test_path_mushroom
		self.test_path_income = args.test_path_income


	def _read_data(self, path, dtype, skip_header=False, with_label=True):
		data = []
		with open(path, 'r', encoding='utf-8') as f:
			file = csv.reader(f, delimiter=',', quotechar='\r')
			if skip_header: next(file, None)  # skip the headers
			for row in file:
				if dtype == 'float': data.append([float(item) for item in row])
				if dtype == 'str': data.append([str(item).strip() for item in row])
		if with_label:
			data = np.asarray(data)
			return list(data[:, :-1]), list(data[:, -1])
		else:
			return data


	def _check_and_display(self, train_x, train_y, test_x, test_y=[]):
		print('>> [Data Loader] Training x data shape:', np.shape(train_x))
		print('>> [Data Loader] Training y data shape:', np.shape(train_y))
		print('>> [Data Loader] Testing x data shape:', np.shape(test_x))
		if len(test_y) != 0: print('>> [Data Loader] Testing y data shape:', np.shape(test_y))
		if len(test_y) != 0: assert np.shape(test_x)[0] == np.shape(test_y)[0]
		assert np.shape(train_x)[0] == np.shape(train_y)[0]


	def _to_different_dtype(self, data):
		new_data = []
		for i, row in enumerate(data):
			new_row = []
			for j, item in enumerate(row):
				try: new_row.append(int(item)) # dtype == int
				except: new_row.append(str(item)) # dtype == str
			new_data.append(new_row)
		return new_data


	def _preprocess_mushroom(self, train_x, test_x):
		encoder = OneHotEncoder(handle_unknown='ignore')
		encoder.fit(train_x)
		train_x = encoder.transform(train_x).toarray()
		test_x = encoder.transform(test_x).toarray()
		return train_x, test_x


	def _preprocess_income(self, train_x, test_x, norm=False):

		#--separate str and int dtype---#
		train_x = self._to_different_dtype(train_x)
		test_x = self._to_different_dtype(test_x)

		#---replace ? with np.nan---#
		train_x = pd.DataFrame([[np.nan if item == '?' else item for item in row] for row in train_x])
		test_x = pd.DataFrame([[np.nan if item == '?' else item for item in row] for row in test_x])

		#---impute missing value---#
		imputer = DataImputer()
		imputer.fit(train_x)
		train_x = imputer.transform(train_x).values
		test_x = imputer.transform(test_x).values
		
		#---split into categorical and continuous---#
		categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
		continuous_features = [0, 2, 4, 10, 11, 12]
		train_x_cat = np.take(train_x, indices=categorical_features, axis=1)
		train_x_con = np.take(train_x, indices=continuous_features, axis=1).astype(np.float64)
		test_x_cat = np.take(test_x, indices=categorical_features, axis=1)
		test_x_con = np.take(test_x, indices=continuous_features, axis=1).astype(np.float64)

		#---transform categocial to one hot---#
		encoder = OneHotEncoder(handle_unknown='ignore')
		encoder.fit(train_x_cat)
		train_x_cat = encoder.transform(train_x_cat).toarray()
		test_x_cat = encoder.transform(test_x_cat).toarray()

		# #---normalize continuous data---#
		if norm:
			normalizer = Normalizer(norm='max', copy=False)
			normalizer.fit(train_x_con)
			train_x_con = normalizer.transform(train_x_con)
			test_x_con = normalizer.transform(test_x_con)

		#---concatenate and split---#
		train_x = np.concatenate((train_x_cat, train_x_con), axis=1)
		test_x = np.concatenate((test_x_cat, test_x_con), axis=1)
		
		
		return train_x, test_x


	def fetch_news(self):
		print('>> [Data Loader] Reading the News dataset...')
		train_x, train_y = self._read_data(self.train_path_news, dtype='float')
		test_x, test_y = self._read_data(self.test_path_news, dtype='float')
		
		self._check_and_display(train_x, train_y, test_x, test_y)
		return train_x, train_y, test_x, test_y


	def fetch_mushroom(self):
		print('>> [Data Loader] Reading the Mushroom dataset...')
		train_x, train_y = self._read_data(self.train_path_mushroom, dtype='str')
		test_x, test_y = self._read_data(self.test_path_mushroom, dtype='str')
		train_x, test_x = self._preprocess_mushroom(train_x, test_x)
		self._check_and_display(train_x, train_y, test_x, test_y)
		return train_x, train_y, test_x, test_y


	def fetch_income(self):
		print('>> [Data Loader] Reading the Income dataset...')
		train_x, train_y = self._read_data(self.train_path_income, dtype='str')
		test_x = self._read_data(self.test_path_income, dtype='str', with_label=False)
		train_x, test_x = self._preprocess_income(train_x, test_x)
		self._check_and_display(train_x, train_y, test_x)
		return train_x, train_y, test_x, None


################
# DATA IMPUTER #
################
class DataImputer(TransformerMixin):

	def __init__(self):
		"""
		Impute missing values.
		- Columns of dtype object are imputed with the most frequent value in column.
		- Columns of other types are imputed with mean of column.
		- Reference: https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
		"""
	def fit(self, X, y=None):
		self.fill = pd.Series([X[c].value_counts().index[0]
			if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
			index=X.columns)
		return self

	def transform(self, X, y=None):
		return X.fillna(self.fill)

