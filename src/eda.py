import pandas as pd

class EDA():
	""" Class to be used for performing all EDA for ML projects """

	def printMissing(self, df):
		cols = df.columns
		print("#"*25)
		for c in cols:
			print('# of missing values in column ', c, ': ', df[c].isnull().sum())
		print("#"*25)


def main():
	df = pd.read_csv('train.csv')
	do_eda = EDA()
	do_eda.printMissing(df)


if __name__ == "__main__":
	main()