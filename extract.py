import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

state_abbr = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
	"HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
	"MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
	"NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
	"SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
state_abbr = [s.lower() for s in state_abbr]

def get_data(filename = None):
	usecols = ['Cusip', 'Yield', 'Name', 'State', 'IssueSize', 'IssueType', 'TradeType', 'IssueSource', 'Coupon', 'Maturity', 'RTG_Moody', 'RTG_SP', 'Price']
	
	#read data
	if filename is None:
		filename = "./data/TMC_020617.csv"

	d = pd.read_csv(filename, usecols = usecols)

	#make column names all lower case
	d.columns = [x.lower() for x in d.columns]
	return d

def clean_data(d):
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

#	print d.shape[0]
#	#pick only entries with tradetype == Sale_to_Customer
#	I = d['tradetype'] != "Sale_to_Customer"
#	d = d.loc[I]
#	print d.shape[0]

	#drop rtg_moody
	d = d.drop(['rtg_moody'], axis = 1)

	#drop rows with missing values for existing data	
	d = d.dropna()

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

	#make names all lower case
	d.name = d.name.apply(lambda x: x.lower())

	return d

def transform_data(d):

	#drop cusip and name for now
	#d = d.drop(['cusip','name','tradetype'], axis = 1)
	d = d.drop(['cusip'], axis = 1)

	#categorical variables are state, issuetype, issuesource
	#use one hot binarization on state variable
	le_state = preprocessing.LabelEncoder().fit(d.state) 
	d.state = le_state.transform(d.state)
	
	le_issuetype = preprocessing.LabelEncoder().fit(d.issuetype)
	d.issuetype = le_issuetype.transform(d.issuetype)	

	le_issuesource = preprocessing.LabelEncoder().fit(d.issuesource)
	d.issuesource = le_issuesource.transform(d.issuesource)

	price = d.pop('price')
	return d, price

def state_abbr_filter(s):
	if s in state_abbr:
		return True
	
	elif 

def build_name_features(d):
	#assumes d is already cleaned

	state_names = list(d.state.unique())
	state_names = [s.lower() for s in state_names].sort()

	#remove state_names from string name if they do occur
	def state_strip(s):
		for state in state_names:
			if state in s:
				s = s.split(state)[-1]
		return s.split()

	L = d.name.apply(state_strip)

	#stop_words = [
		
	#split words of each name into separate strings ('-', ' ', '/'), indexed by column
	L = d.name.str.split(r'[\-/ ]', expand = True)

	#eliminate null/None values and concatenate columns into single column of words
	S = pd.concat([L[j].dropna() for j in xrange(len(L.columns))], axis = 0)
	S = S[S.apply(lambda x: len(x)) > 1] #filter out strings with only one character

	longer_name_features = S[S.apply(lambda x: len(x) > 4)].value_counts()[:250]
	shorter_name_features = list(set(S.value_counts()[:50].index.tolist())-set(state_abbr))

	combined_name_features = longer_name_features + shorter_name_features
	
	d_aug = pd.DataFrame(np.zeros((d.shape[0], len(combined_name_features)), dtype = np.int))
	for j in xrange(len(combined_name_features)):
		d_aug[j] = d.name.apply(lambda x: combines_name_features[j] in x).astype(np.int)
	
	pd.pop(d.name) #remove name column after adding augmented features

	return pd.concat([d, d_aug], axis = 1)
	

	
	
	

def count_unique_occurences(d):
	"""
	for state, issuetype, issuesource, name: show frequency of each category
	"""

	print "state: "
	print d.state.value_counts()

	print "issuetype: "
	print d.issuetype.value_counts()
	
	print "issuesource: "
	print d.issuesource.value_counts()

	print "name: "
	

	


if __name__ == "__main__":
	main()
