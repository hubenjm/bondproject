import extract
import pandas as pd
import argparse

def main():
	parser = argparse.ArgumentParser(description = "extract price difference data from original data set and save to file.")
	parser.add_argument("-mm", "--minmarketvalue", nargs='?', default = 0, type=float)
	parser.add_argument("-rl", "--ratinglimit", nargs='?', default=22, choices = range(1,23), type=int)
	parser.add_argument("-tt", "--tradetype", nargs='?', default='purchase', choices = ['purchase', 'sell', 'sequence'], type = str)
	parser.add_argument("-c", "--usecolumns", nargs='?', default='default', choices = ['default', 'alt'], type = str)
	parser.add_argument("-i", "--inputfile", nargs='?', default='./data/TMC_020617.csv', type=str)
	parser.add_argument("-o", "--outputfile", nargs='?', default='./data/price_change_data.csv', type=str)
	args = parser.parse_args()

	d = extract.get_data(filename = args.inputfile, usecols = args.usecolumns)

	if args.usecolumns == 'alt':
		#clean up some inconsistencies
		d.state = d.state.apply(lambda x: x.title())
		d.issuetype = d.issuetype.apply(lambda x: x.upper())
		d.issuesource = d.issuesource.apply(lambda x: x.upper())
		d.loc[d.tradetype == 'Customer sold', 'tradetype'] = 'Purchase_from_Customer'
		d.loc[d.tradetype == 'Customer bought', 'tradetype'] = 'Sale_to_Customer'

	d = extract.clean_data(d)

	print("finished loading and cleaning data")
	print("generating price difference data set...")
	if args.tradetype == 'sequence':
		fixed_tradetype = None
	else:
		fixed_tradetype = args.tradetype
		
	s = extract.compile_price_change_data(d, fixed_tradetype = fixed_tradetype, tradetype_dict = {'purchase': 'Sale_to_Customer', 'sell': 'Purchase_from_Customer'}, marketvalue_min = args.minmarketvalue, rtg_limit = args.ratinglimit)

	s.to_csv(args.outputfile)

if __name__ == "__main__":
	main()
