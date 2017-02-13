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

def get_data_simple(filename = None):
	d = get_data(filename)
	d = clean_data(d)

	d_state = build_state_features(d, 15)
	d_simple_text = build_other_text_features(d)
	d = pd.concat([d.drop(['issuetype', 'issuesource', 'tradetype', 'name', 'state'], axis = 1), d_simple_text, d_state], axis = 1)
	return d

def get_data(filename = None):
	usecols = ['Yield', 'Name', 'State', 'IssueSize', 'IssueType', 'TradeType', 'IssueSource', 'Coupon', 'Maturity', 'RTG_Moody', 'RTG_SP', 'Price']
	
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

	#drop rtg_moody
	d = d.drop(['rtg_moody'], axis = 1)

	#drop rows with missing values for existing data	
	d = d.dropna()
	d.index = range(d.shape[0]) #reset indices on rows to be consecutive after dropping NAs
	
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

def get_text_features(d):
	pass

def state_abbr_filter(s):
	if s in state_abbr:
		return ''

	for code in state_abbr:
		if " " + code + " " in s:
			return ''.join(s.split(" " + code + " "))

		if " " + code in s[-3:]:
			return s[:-3]

		if code + " " in s[:3]:
			return s[3:]
		
	return s

def build_name_features(d):
	#assumes extract.clean_data(d) has already been performed

	state_names = list(d.state.unique()) + ['massachusets']
	state_names = [s.lower() for s in state_names]
	state_names.sort()

	#remove state_names from string name if they do occur
	#state names always seem to occur at beginning of name
	def state_strip(s):
		for state in state_names:
			if state in s:
				s = ''.join(s.split(state))
		return s.strip()

	#apply state_name_strip and state_abbr_strip
	#split words of each name into separate strings ('-', ' ', '/'), indexed by column
	L = d.name.apply(state_strip).apply(state_abbr_filter).str.split(r'[\-/ ]', expand = True)
		
	#eliminate null/None values and concatenate columns into single column of words
	S = pd.concat([L[j].dropna() for j in xrange(len(L.columns))], axis = 0)
	S = S[S.apply(lambda x: len(x)) > 1] #drop strings with only one character

	longer_name_features = S[S.apply(lambda x: len(x) > 4)].value_counts()[:250].index.tolist()
	shorter_name_features = list(set(S.value_counts()[:50].index.tolist())-set(state_abbr))

	combined_name_features = longer_name_features + shorter_name_features
	
	d_aug = pd.DataFrame(np.zeros((d.shape[0], len(combined_name_features)), dtype = np.int))
	for j in xrange(len(combined_name_features)):
		d_aug[j] = d.name.apply(lambda x: combined_name_features[j] in x).astype(np.int)
	
	return d_aug
	
def build_state_features(d, num_states = None):
	if num_states is None:
		return pd.get_dummies(d.state)
	else:
		d_aug = pd.get_dummies(d.state)
		assert num_states < d.state.unique().size and num_states > 0
		features = d.state.value_counts()[:num_states].index.tolist() #pick out top num_states states from d.state Series
		other_states = list(set(d.state.unique()) - set(features))
		d_aug['otherstates'] = d_aug.loc[:,other_states].sum(axis = 1)
		for state in other_states:
			d_aug.pop(state)

		return d_aug

def build_other_text_features(d):
	d_issuetype = pd.get_dummies(d.issuetype)
	d_issuesource = pd.get_dummies(d.issuesource)
	d_tradetype = d.tradetype.apply(lambda x: str(x) == 'Sale_to_Customer').astype(np.int)
	return pd.concat([d_issuetype, d_issuesource, d_tradetype], axis = 1)

if __name__ == "__main__":
	main()
