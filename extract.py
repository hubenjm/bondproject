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
	#usecols = ['Cusip', 'Yield', 'Name', 'State', 'IssueSize', 'IssueType', 'TradeType', 'IssueSource', 'Coupon', 'Maturity', 'RTG_Moody', 'RTG_SP', 'Price', 'TradeID']
	usecols = ['TradeId', 'Cusip', 'Amount', 'Price', 'Yield', 'TradeDate', 'TradeType', 'Name', 'State', 'RTG_Moody', 'RTG_SP', 'Coupon', 'Maturity',
	'IssueSize', 'IssueType', 'IssueSource', 'BidCount']
	#read data
	if filename is None:
		filename = "./data/TMC_020617.csv"

	d = pd.read_csv(filename, usecols = usecols)

	#make column names all lower case
	d.columns = [x.lower() for x in d.columns]

	return d

def clean_data(d):
	#remove entries with bad cusip
	def cusip_filter(s):
		if 'E+' in s:
			return False
		else:
			return True

	d.loc[:] = d.loc[d.cusip.apply(cusip_filter)]

	#merge RTG_Moody and RTG_SP columns based on missing values
	moody_to_sp_dict = {'Aaa':'AAA', 'Aa1':'AA+', 'Aa2':'AA',
		'Aa3':'AA-', 'A1':'A+', 'A2':'A', 'A3':'A-',
		'Baa1':'BBB+', 'Baa2':'BBB', 'Baa3':'BBB-',
		'Ba1':'BB+', 'Ba2':'BB', 'Ba3':'BB-',
		'B1':'B+', 'B2':'B', 'B3':'B-',
		'Caa1':'CCC+', 'Caa2':'CCC', 'Caa3':'CCC-',
		'Ca':'CC', 'C':'C', '/':'D'}

	sp_to_moody_dict = {'AAA':'Aaa', 'AA+':'Aa1', 'AA':'Aa2',
		'AA-':'Aa3', 'A+':'A1', 'A':'A2', 'A-':'A3',
		'BBB+':'Baa1', 'BBB':'Ba2', 'BBB-':'Baa3',
		'BB+':'Ba1','BB':'Ba2','BB-':'Ba3','B+':'B1',
		'B':'B2','B-':'B3','CCC+':'Caa1','CCC':'Caa2',
		'CCC-':'Caa3','CC':'Ca','C':'C','D':'/'}

	f_ms = lambda x: moody_to_sp_dict[x]
	f_sm = lambda x: sp_to_moody_dict[x]

	#correct a typo in rating data entry
	d.loc[d.rtg_moody == 'Ca1', 'rtg_moody'] = 'Caa1' 

	#pick out rows where RTG_SP is null but RTG_Moody is not
	#set 'rtg_sp' equal to that value
	I = pd.isnull(d['rtg_sp']) & ~pd.isnull(d['rtg_moody'])
	d.loc[I, 'rtg_sp'] = d.loc[I, 'rtg_moody'].apply(f_ms)

	#pick out rows where RTG_Moody is null but RTG_SP is not null
	I = ~pd.isnull(d['rtg_sp']) & pd.isnull(d['rtg_moody'])
	d.loc[I, 'rtg_moody'] = d.loc[I, 'rtg_sp'].apply(f_sm)

	#remove entries of d where both rtg_moody and rtg_sp are null
	d = d[~pd.isnull(d.rtg_moody)]

	#convert rtg_sp values to numerical values based on standard ranking
	sp_rating_dict = {'AAA':1, 'AA+':2, 'AA':3,
		'AA-':4, 'A+':5, 'A':6, 'A-':7,
		'BBB+':8, 'BBB':9, 'BBB-':10,
		'BB+':11, 'BB':12, 'BB-':13, 'B+':14,
		'B':15, 'B-':16, 'CCC+':17, 'CCC':18,
		'CCC-':19, 'CC':20, 'C':21, 'D':22}
	g_sp = lambda x: sp_rating_dict[x]
	d.loc[:, 'rtg_sp'] = d['rtg_sp'].apply(g_sp)

#	#convert rtg_moody values to numerical values based on standard ranking
	moody_rating_dict = {'Aaa':1, 'Aa1':2, 'Aa2':3,
		'Aa3':4, 'A1':5, 'A2':6, 'A3':7,
		'Baa1':8, 'Baa2':9, 'Baa3':10,
		'Ba1':11, 'Ba2':12, 'Ba3':13, 'B1':14,
		'B2':15, 'B3':16, 'Caa1':17, 'Caa2':18,
		'Caa3':19, 'Ca':20, 'C':21, '/':22}

	g_moody = lambda x: moody_rating_dict[x]
	d.loc[:,'rtg_moody'] = d['rtg_moody'].apply(g_moody)

	#compute minimum of RTG_SP and RTG_MOODY
	d['rtg'] = d[["rtg_moody", "rtg_sp"]].min(axis=1)

	#drop rtg_moody
	d = d.drop(['rtg_moody', 'rtg_sp'], axis = 1)

	#drop entries with negative yield
 	d = d[d['yield'] >= 0]
	
	#adjust maturity date data to be days past minimum date in data set
	d['maturity'] = pd.to_datetime(d['maturity'])
	min_maturity_date = d['maturity'].min()
	d['maturity'] = (d['maturity'] - min_maturity_date)/np.timedelta64(1,'D')
		
	#add column for time since minimum trade date
	d['dtradedate'] = pd.to_datetime(d['tradedate'])
	min_tradedate = d['dtradedate'].min()
	d['dtradedate'] = (d['dtradedate'] - min_tradedate)/np.timedelta64(1, 'D')

	#make names all lower case
	d.name = d.name.apply(lambda x: x.lower())
	d.index = range(d.shape[0])

	return d

#def transform_data(d):

#	#drop cusip and name for now
#	#d = d.drop(['cusip','name','tradetype'], axis = 1)
#	d = d.drop(['cusip'], axis = 1)

#	#categorical variables are state, issuetype, issuesource
#	#use one hot binarization on state variable
#	le_state = preprocessing.LabelEncoder().fit(d.state) 
#	d.state = le_state.transform(d.state)
#	
#	le_issuetype = preprocessing.LabelEncoder().fit(d.issuetype)
#	d.issuetype = le_issuetype.transform(d.issuetype)	

#	le_issuesource = preprocessing.LabelEncoder().fit(d.issuesource)
#	d.issuesource = le_issuesource.transform(d.issuesource)

#	price = d.pop('price')
#	return d, price

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
	state_names = [s.lower() for s in state_names].sort()

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

	longer_name_features = S[S.apply(lambda x: len(x) > 4)].value_counts()[:250]
	shorter_name_features = list(set(S.value_counts()[:50].index.tolist())-set(state_abbr))

	combined_name_features = longer_name_features + shorter_name_features
	
	d_aug = pd.DataFrame(np.zeros((d.shape[0], len(combined_name_features)), dtype = np.int))
	for j in xrange(len(combined_name_features)):
		d_aug[j] = d.name.apply(lambda x: combines_name_features[j] in x).astype(np.int)
	
	pd.pop(d.name) #remove name column after adding augmented features

	return pd.concat([d, d_aug], axis = 1)
	
def build_state_features(d, num_states = None):
	if num_states is not None:
		assert num_states <= d.state.unique().size and num_states > 0
		features = d.state.value_counts()[:num_states].index.tolist() + ['Other']
		for j in xrange(len(features) - 1):
			d_aug[j] = d.name.apply(lambda x: x == features[j]).astype(np.int)
	
	else:
		features = d.state.value_counts().index.tolist()

	d_aug = pd.DataFrame(np.zeros((d.shape[0], len(features)), dtype = np.int), columns = features)
	
	for j in xrange(len(features)):
		d_aug[j] = d.name.apply(lambda x: x == features[j]).astype(np.int)
	
	return pd.concat([d, d_aug], axis = 1)


def compile_price_change_data(d):
	unique_cusip = list(d.cusip.unique())
	dindex_set = [] #for keeping track of which entry of original data set d to parse from when filling in price change table
					#each row of price change set will correspond to a specific bond at a specific tradedate and a specific dtradedate
	dprice_data = []
	holdtime_data = [] #keeps track of dprice and holdtime for each price change event

	for i, s in enumerate(unique_cusip):
		#get all data points for given cusip
		d_slice = d.loc[d.cusip == s].sort(columns = ['dtradedate', 'tradeid'], ascending = [1,1])
		d_purchase = d_slice[d_slice.tradetype == 'Sale_to_Customer']	
		d_sell = d_slice[d_slice.tradetype == 'Purchase_from_Customer']
	
		if d_sell.size == 0 or d_purchase.size == 0:
			continue #no data points to compare for price changes
		
		else:
			#go through d_purchase sequentially and slice from d_sell for dtradedate values greater than current dtradedate
			for j in d_purchase.index:
				d_entry = d_purchase.loc[j,:]
				dp = d_sell.loc[d_sell.dtradedate > d_entry.dtradedate].price.values
				if dp.shape[0] > 0:
					dp -= d_entry.price
					dindex_set += dp.shape[0]*[j] #add row j of d, dp.shape[0] times
					holdtime = d_sell.loc[d_sell.dtradedate > d_entry.dtradedate].dtradedate - d_entry.dtradedate
					dprice_data += list(dp)
					holdtime_data += list(holdtime)

		print("extract.compile_price_change_data: finished cusip {} of {}".format(i, len(unique_cusip)))

		
	#create data frame that includes all dprice and holdtime data, set its index to dindex_set and perform inner join with d
	t = pd.DataFrame(np.vstack((dprice_data, holdtime_data)).T, columns = ['dprice', 'holdtime'])
	t.index = dindex_set
	
	joined_data = pd.concat([d, t], axis=1, join='inner')
	return joined_data

		

if __name__ == "__main__":
	main()
