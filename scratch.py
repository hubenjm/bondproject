import pandas as pd
import word2vec

def main():

	usecols = ['Cusip', 'Yield', 'Cusip.1', 'Name', 'State', 'IssueSize', 'IssueType', 'IssueSource', 'Coupon', 'Maturity', 'RTG_Moody', 'RTG_SP']

	#read data
	filename = "./data/TMC_020617.csv"
	d_raw = pd.read_csv(filename, usecols = usecols)
	d = d_raw.IssueSource.as_matrix().ravel()
	np.savetxt('./issuesourcewords.txt', d, fmt = '%s')
	word2vec.word2phrase('./issuesourcewords.txt', './issuesourcephrases.txt', min_count = 1, verbose = True)
	word2vec.word2vec('issuesourcephrases.txt', 'issuesource.bin', min_count = 1, size = 100, verbose = True)
	model = word2vec.load('issuesource.bin')

	
	
	model.generate_response(indices, metrics).tolist()
	A = model.vectors
	model.vocab
	indices, metrics = model.cosine('TAX')
	model.generate_response(indices, metrics).tolist()


		
		
