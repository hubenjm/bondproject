import extract
import pandas as pd

def main():
	d = extract.get_data()
	d = extract.clean_data(d)

	print("finished loading and cleaning data")
	print("generating price difference data set...")
	s = extract.compile_price_change_data(d)
	s.to_csv("price_change_data.csv")

if __name__ == "__main__":
	main()
