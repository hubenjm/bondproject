import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

state_abbr = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
	"HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
	"MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
	"NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
	"SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
state_abbr = [s.lower() for s in state_abbr]

#['TradeId', 'Cusip', 'Amount', 'Price', 'Yield', 'TradeDate', 'TradeType', 'Name', 'State', 'RTG_Moody', 'RTG_SP', 'Coupon', 'Maturity',
#	'IssueSize', 'IssueType', 'IssueSource', 'BidCount']

def get_data_simple(filename = None):
	d = get_data(filename)
	d = clean_data(d)

	d_state = build_state_features(d, 15)
	d_simple_text = build_other_text_features(d)
	d = pd.concat([d.drop(['issuetype', 'issuesource', 'tradetype', 'name', 'state', 'bidcount', 'cusip', 'tradeid'], axis = 1), d_simple_text, d_state], axis = 1)
	return d

def compute_min_rtg(d):
	"""
	take minimum of rtg_moody and rtg_sp if both exist, take one if only one is provided, and leave as NA if neither exists
	"""
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

	#convert rtg_moody values to numerical values based on standard ranking
	moody_rating_dict = {'Aaa':1, 'Aa1':2, 'Aa2':3,
		'Aa3':4, 'A1':5, 'A2':6, 'A3':7,
		'Baa1':8, 'Baa2':9, 'Baa3':10,
		'Ba1':11, 'Ba2':12, 'Ba3':13, 'B1':14,
		'B2':15, 'B3':16, 'Caa1':17, 'Caa2':18,
		'Caa3':19, 'Ca':20, 'C':21, '/':22}

	g_moody = lambda x: moody_rating_dict[x]
	d.loc[:,'rtg_moody'] = d['rtg_moody'].apply(g_moody)

	#compute minimum of RTG_SP and RTG_MOODY
	d.loc[:,'rtg'] = d[["rtg_moody", "rtg_sp"]].max(axis=1)

	#drop rtg_moody
	d.pop('rtg_moody')
	d.pop('rtg_sp')

	return d
	
def get_data(filename = "./data/TMC_020617.csv", usecols = 'default'):

	if usecols == 'default':
		usecols = ['TradeId', 'Cusip', 'Amount', 'Price', 'Yield', 'TradeDate', 'TradeType', 'Name', 'State', 'RTG_Moody', 'RTG_SP', 'Coupon', 'Maturity', 'IssueSize', 'IssueType', 'IssueSource', 'BidCount']
	elif usecols == 'alt':
		usecols = ['cusip', 'price', 'yield', 'amount', 'tradedate', 'tradetype', 'name', 'state', 'RTG_Moody', 'RTG_SP', 'coupon', 'maturity', 'issuetype', 'issuesource']

	#read data
	d = pd.read_csv(filename, usecols = usecols)

	#make column names all lower case
	d.columns = [x.lower() for x in d.columns]

	return d

def clean_data(d, reference_date = '01/01/2017'):
	date_i = pd.to_datetime(reference_date)
	#remove entries with bad cusip
	def cusip_filter(s):
		if 'E+' in s:
			return False
		else:
			return True

	d.loc[:] = d.loc[d.cusip.apply(cusip_filter)]

	d = compute_min_rtg(d)

	#drop entries with negative yield
 	d = d[d['yield'] >= 0]
	
	#add column for time since minimum trade date
	d['tradedate'] = pd.to_datetime(d['tradedate'])
	d['maturity'] = pd.to_datetime(d['maturity'])

	d['maturity'] = (d['maturity'] - d['tradedate'])/np.timedelta64(1,'D')
	d['tradedate'] = (d['tradedate'] - date_i)/np.timedelta64(1, 'D')

	#clean up trade type
	def tradetype_filter(s):
		if s.lower() == 'customer sold':
			return "Purchase_from_Customer"
		elif s.lower() == 'customer bought':
			return "Sale_to_Customer"
		else:
			return s
		
	d.tradetype = d.tradetype.apply(tradetype_filter)

	#make names all lower case
	d.name = d.name.apply(lambda x: x.lower())
	d.index = range(d.shape[0])

	#add prefix to Issue Source and Issue Type to eliminate potential collisions
	if 'issuesource' in d.columns:
		d.issuesource = d.issuesource.apply(lambda x: "ISSUE SOURCE: " + str(x))
	if 'issuetype' in d.columns:
		d.issuetype = d.issuetype.apply(lambda x: "ISSUE TYPE: " + str(x))

	return d

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

def build_name_features(d, num_general_words = 50, num_long_words = 250, long_word_length = 5):
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

	longer_name_features = S[S.apply(lambda x: len(x) >= long_word_length)].value_counts()[:num_long_words].index.tolist()
	shorter_name_features = list(set(S.value_counts()[:num_general_words].index.tolist())-set(state_abbr))

	combined_name_features = longer_name_features + shorter_name_features
	
	d_aug = pd.DataFrame(np.zeros((d.shape[0], len(combined_name_features)), dtype = np.int))
	for word in set(combined_name_features):
		d_aug[word] = d.name.apply(lambda x: word in x).astype(np.int)
	
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

def build_other_text_features(d, features = ['issuetype', 'issuesource'], tradetype = True):
	d_other = {}
	for j in range(len(features)):
		d_other[features[j]] = pd.get_dummies(d[features[j]])
#	d_issuetype = pd.get_dummies(d.issuetype)
#	d_issuesource = pd.get_dummies(d.issuesource)

	if tradetype:
		d_tradetype = d.tradetype.apply(lambda x: str(x) == 'Sale_to_Customer').astype(np.int)
		return pd.concat([d_other[j] for j in d_other] + [d_tradetype], axis = 1)
	else:
		return pd.concat([d_other[j] for j in d_other], axis = 1)

def compile_price_change_data(d, fixed_tradetype = None, tradetype_dict = {'purchase': 'Sale_to_Customer', 'sell': 'Purchase_from_Customer'}, marketvalue_min = 0.0, rtg_limit = 22):
	"""
	marketvalue_min, specifies minimum marketvalue amount to consider for trade filters (e.g. 20000)
	"""
	assert fixed_tradetype in ['purchase', 'sell'] or fixed_tradetype is None

	#tradetype_dict = {'purchase': 1, 'sell': 0}
	unique_cusip = list(d.cusip.unique())
	dindex_set_1 = [] #d_index for first trade event
	dindex_set_2 = [] #d_index for second trade event
	tradedate_2 = []

	dprice_data = []
	holdtime_data = [] #keeps track of dprice and holdtime for each price change event

	#filter out all bonds with less than marketvalue_min before proceeding
	if 'amount' in d.columns:
		d_filter = d.loc[d.amount*d.price/100.0 >= marketvalue_min, :]
	else:
		d_filter = d

	for s in tqdm(unique_cusip):
		#get all data points for given cusip
		I = (d_filter.cusip == s) & (d_filter.rtg <= rtg_limit)
		
		if 'tradeid' in d.columns:
			d_slice = d_filter.loc[I].sort_values(by = ['tradedate', 'tradeid'], ascending = [True, True])
		else:
			d_slice = d_filter.loc[I].sort_values(by = ['tradedate'], ascending = [True])

		if fixed_tradetype is not None:
			d_slice = d_slice.loc[d_slice.tradetype == tradetype_dict[fixed_tradetype], :]
			if d_slice.size == 0:
				continue
			else:
				for j in d_slice.index:
					d_entry = d_slice.loc[j,:]
					J = d_slice.tradedate > d_entry.tradedate
					dp = d_slice.loc[J].price.values
					if dp.shape[0] > 0:
						dp -= d_entry.price
						dindex_set_1 += dp.shape[0]*[j] #add row j of d, dp.shape[0] times
						dindex_set_2 += list(d_slice[J].index.values)
						holdtime = d_slice.loc[J].tradedate - d_entry.tradedate
						dprice_data += list(dp)
						holdtime_data += list(holdtime)
						
						tradedate_2 += list(d_slice.loc[J].tradedate.values)

		else: #assume we want to collect buy->sell pairs
			#'Sale_to_Customer': 1, 'Purchase_from_Customer': 0
			d_purchase = d_slice[d_slice.tradetype == tradetype_dict['purchase']]
			d_sell = d_slice[d_slice.tradetype == tradetype_dict['sell']]
		
			if d_sell.size == 0 or d_purchase.size == 0:
				continue #no data points to compare for price changes
		
			else:
				#go through d_purchase sequentially and slice from d_sell for dtradedate values greater than current dtradedate
				for j in d_purchase.index:
					d_entry = d_purchase.loc[j,:]
					J = d_sell.tradedate > d_entry.tradedate
					dp = d_sell.loc[J].price.values
					if dp.shape[0] > 0:
						dp -= d_entry.price
						dindex_set_1 += dp.shape[0]*[j] #add row j of d, dp.shape[0] times
						dindex_set_2 += d_sell[J].index.values
						holdtime = d_sell.loc[J].tradedate - d_entry.tradedate
						dprice_data += list(dp)
						holdtime_data += list(holdtime)

						tradedate_2 += list(d_sell.loc[J].tradedate.values)
		
	#create data frame that includes all dprice and holdtime data, set its index to dindex_set and perform inner join with d
	t = pd.DataFrame(np.vstack((dprice_data, holdtime_data)).T, columns = ['dprice', 'holdtime'])
	t.loc[:, 'd_index_1'] = dindex_set_1
	t.loc[:, 'd_index_2'] = dindex_set_2
	t.loc[:, 'tradedate_2'] = tradedate_2

	d_filter.loc[:,'d_index_1'] = d_filter.index

	#joined_data = pd.concat([d_filter, t], axis=1, join='inner')
#	joined_data['d_index_1'] = joined_data.index.values
	joined_data = t.merge(d_filter, how='left', on='d_index_1')
	return joined_data

if __name__ == "__main__":
	main()
