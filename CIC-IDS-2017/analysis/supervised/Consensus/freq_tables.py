print("""
*************************************************************
Frquency Tables Generator and One Hot Encoding transformation
FM, Feb 2019, http://cn.tuwien.ac.at
*************************************************************
	""")

#############################################################
# script name: freq_tables.py
version = "0.0.1"
#############################################################

import sys
import pandas as pd
import shutil
import os

#############################################################

output_folder 	  = "frequency_tables"
column_exceptions = ['Attack', 'Label']

# defaults
mode       		  = "stat"
input_file 		  = ""
topn      		  = 10
threshold  		  = 95
out_file   		  = "freq_tables_data.csv"

#############################################################

def _get_dummies(column, threshold):
	column 		  = column.copy()
	counts 		  = pd.value_counts(column) / column.shape[0] * 100
	mask 		  = column.isin(counts[counts >= threshold].index)
	column[~mask] = "others"
	tmp = pd.get_dummies(column, prefix=column.name)
	try:
		return tmp.drop('{}_others'.format(column.name), 1) # per default drop the others column to overcome the k-1 trap
	except:
		return tmp


def main():
	print("Reading {}".format(input_file))
	data = pd.read_csv(input_file)

	if os.path.exists(output_folder):
	    shutil.rmtree(output_folder)
	os.makedirs(output_folder)

	print("Computing the frequency tables for {} feature...".format(len(data.columns)))
	print("*********************************")

	to_convert = []
	for col in data.columns:
		freq = (data[col].value_counts() / data.shape[0] * 100).rename_axis('unique_values').reset_index(name='counts').head(topn)
		freq['Accumulation %'] = freq['counts'].cumsum(axis = 0)
		freq.to_csv("{}/{}.csv".format(output_folder, col))
		acc  = float(freq.tail(1)['Accumulation %'])
		thr  = float(freq.tail(1)['counts'])

		if mode.lower() == "ohe" and acc >= threshold and col not in column_exceptions:
			statement = "top {}\t for {:50s} represent {:.2f} of the data \t {}".format(freq.shape[0], col, acc, "Convert?(yes:enter, no:any character+enter)")
			try:
				choice = input(statement)
			except:
				choice = raw_input(statement) # compatibility with python2
			if choice =="":
				tmp  = _get_dummies(data[col], thr)
				data = pd.concat([data, tmp], axis=1)
				data = data.drop(col, axis=1)
				to_convert.append(col)
		else:
			print("top {}\t for {:50s} represent {:.2f} of the data".format(freq.shape[0], col, acc))

		
	if mode.lower() == "ohe":
		print("*********************************")
		print("The following features were converted into dummies:")
		print(to_convert)
		print("*********************************")

		print("Saving...")
		data.to_csv(out_file, index=False)

	print("Finished without any errors... Exiting")

if __name__ == "__main__":
	try:
		mode       = str(sys.argv[1])
		input_file = str(sys.argv[2])
		topn       = int(sys.argv[3])
		threshold  = int(sys.argv[4])
		out_file   = str(sys.argv[5])
	except:
		print("""
Usage:   > python freq_tables.py <mode> <data> <top-n> <threshold> <output>
	<mode>       stat: gives frequency tables of all features and save them into an external folder
			     ohe : same as stat but perform also one hot encoding of the top n values
	<data>       csv file holding the raw data
	<top-n>      the number of top values to be considered in the frequency tables and the dummy mapping
	<threshold>  for the one hot encoding cosider only those features where the <top-n> represent more than the threshold
	<output>     the output file (csv format)

Example: > python freq_tables.py ohe input.csv 5 98 output.csv
	show statistics for the top five distinct values and map those who reaches 98\% representation only by 5 top

	NOTE: even if you are using the stat mode, please provide an output file (it will not be created)
			""")
		exit(1)
	main()