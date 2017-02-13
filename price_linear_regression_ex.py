import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.kernel_ridge import KernelRidge

def main():
	usecols = ['Cusip', 'Yield', 'Name', 'State', 'IssueSize', 'IssueType', 'IssueSource', 'Coupon', 'Maturity', 'RTG_Moody', 'RTG_SP', 'Price']

	#read data
	filename = "./data/TMC_020617.csv"
	d = pd.read_csv(filename, usecols = usecols)
	#make column names all lower case
	d.columns = [x.lower() for x in d.columns]
	
	#merge RTG_Moody and RTG_SP columns based on missing values
	moody_to_sp_dict = {'Aaa':'AAA', 'Aa1':'AA+', 'Aa2':'AA',
		'Aa3':'AA-', 'A1':'A+', 'A2':'A', 'A3':'A-',
		'Baa1':'BBB+', 'Baa2':'BBB', 'Baa3':'BBB-',
		'Ba1':'BB+', 'Ba2':'BB', 'Ba3':'BB-', 'B1':'B+',
		'B2':'B', 'B3':'B-', 'Caa1':'CCC+', 'Caa2':'CCC',
		'Caa3':'CCC-', 'Ca':'CC', 'C':'C', '/':'D'}

	f = lambda x: moody_to_sp_dict[x]

	#correct a typo in rating data entry
	d.loc[d.rtg_moody == 'Ca1', 'rtg_moody'] = 'Caa1' 

	#pick out rows where RTG_SP is null but RTG_Moody is not
	I = pd.isnull(d['rtg_sp']) & ~pd.isnull(d['rtg_moody'])
	d.loc[I,'rtg_sp'] = d.loc[I,('rtg_moody')].apply(f)

	#drop cusip and name for now
	d = d.drop(['cusip','name','rtg_moody'], axis = 1)

	#drop rows with missing values for existing data	
	d = d.dropna()
	price = d.pop('price')

	#convert rtg_sp values to numerical values based on standard ranking
	sp_rating_dict = {'AAA':1, 'AA+':2, 'AA':3,
		'AA-':4, 'A+':5, 'A':6, 'A-':7,
		'BBB+':8, 'BBB':9, 'BBB-':10,
		'BB+':11, 'BB':12, 'BB-':13, 'B+':14,
		'B':15, 'B-':16, 'CCC+':17, 'CCC':18,
		'CCC-':19, 'CC':20, 'C':21, 'D':22}
	g = lambda x: sp_rating_dict[x]
	d['rtg_sp'] = d['rtg_sp'].apply(g)
	
	#adjust maturity date data to be days past minimum date in data set
	d['maturity'] = pd.to_datetime(d['maturity'])
	min_maturity_date = d['maturity'].min()
	d['maturity'] = (d['maturity'] - min_maturity_date)/np.timedelta64(1,'D')
	
	#categorical variables are state, issuetype, issuesource
	le_state = preprocessing.LabelEncoder().fit(d.state) 
	d.state = le_state.transform(d.state)
	
	le_issuetype = preprocessing.LabelEncoder().fit(d.issuetype)
	d.issuetype = le_issuetype.transform(d.issuetype)	

	le_issuesource = preprocessing.LabelEncoder().fit(d.issuesource)
	d.issuesource = le_issuesource.transform(d.issuesource)

	print d.issuesource.unique().size
	print price.min(), price.max()

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

#	kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5, 
#		param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})

#	kr.fit(d_train, p_train)
#	p_predict = kr.predict(d_test)
#	print("Mean squared error: {}".format(np.mean((p_predict - p_test)**2)))

	


if __name__ == "__main__":
	main()
