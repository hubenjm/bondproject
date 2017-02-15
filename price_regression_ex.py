import numpy as np
import pandas as pd
import extract
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def main():
	d = extract.get_data()
	d = extract.clean_data(d)
	usecols = ['yield', 'state', 'issuesize', 'issuetype', 'issuesource', 'coupon', 'maturity', 'rtg', 'price']
	d = d.loc[:, usecols]

	#drop rows with missing values for existing data	
	d = d.dropna()
	price = d.pop('price')

	#categorical variables are state, issuetype, issuesource
	le_state = preprocessing.LabelEncoder().fit(d.state) 
	d.state = le_state.transform(d.state)
	
	le_issuetype = preprocessing.LabelEncoder().fit(d.issuetype)
	d.issuetype = le_issuetype.transform(d.issuetype)	

	le_issuesource = preprocessing.LabelEncoder().fit(d.issuesource)
	d.issuesource = le_issuesource.transform(d.issuesource)

	#scale variables
	d = StandardScaler().fit_transform(d)

	#split into test and training parts
	d_train, d_test, p_train, p_test = train_test_split(d, price, test_size=0.20, random_state=13)

	print("starting regression...")
	#Regression Part:
	regr = LinearRegression()
	regr.fit(d_train, p_train)

	# The coefficients
	print "Coefficients: \n", regr.coef_
	# The mean squared error
	print("Mean squared error: {}".format(np.mean((regr.predict(d_test) - p_test)**2)))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: {}'.format(regr.score(d_test, p_test)))


if __name__ == "__main__":
	main()
