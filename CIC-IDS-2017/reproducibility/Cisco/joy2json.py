#################################################
#
#
# Name        : joy2json.py
# Description : This script takes a (special) json output from the joy tool and outputs a valid json file
# Author      : Fares Meghdouri
#
#
#################################################

import ast
import json
import sys
import gzip

#################################################
files = ['Friday_original','Monday_original','Tuesday_original','Thursday_original','Wednesday_original']
#################################################

def main():
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []

    errors = 0
    print(">> Loading the data")
    #for input_name in file_list:
    with gzip.open(files[0]) as openfileobject:
        for line in openfileobject:
            try:
                data1.append(json.loads(line.decode("ascii").replace(',,',',')))
            except:
                errors+=1
                pass
    print("finished decoding, {} errors detected".format(errors))
            
    data1.pop(0)

    with gzip.open(files[1]) as openfileobject:
        for line in openfileobject:
            try:
                data2.append(json.loads(line.decode("ascii").replace(',,',',')))
            except:
                errors+=1
                pass
    print("finished decoding, {} errors detected".format(errors))
            
    data2.pop(0)

    with gzip.open(files[2]) as openfileobject:
        for line in openfileobject:
            try:
                data3.append(json.loads(line.decode("ascii").replace(',,',',')))
            except:
                errors+=1
                pass
    print("finished decoding, {} errors detected".format(errors))
            
    data3.pop(0)

    with gzip.open(files[3]) as openfileobject:
        for line in openfileobject:
            try:
                data4.append(json.loads(line.decode("ascii").replace(',,',',')))
            except:
                errors+=1
                pass
    print("finished decoding, {} errors detected".format(errors))
            
    data4.pop(0)

    with gzip.open(files[4]) as openfileobject:
        for line in openfileobject:
            try:
                data5.append(json.loads(line.decode("ascii").replace(',,',',')))
            except:
                errors+=1
                pass
    print("finished decoding, {} errors detected".format(errors))
            
    data5.pop(0)

    data = data1 + data2 + data3 + data4 + data5


    print(">> Saving the data")
    with open("all.json", 'w') as fout:
        json.dump(data, fout)
    print(">> Done")
# end def main

if __name__ == "__main__":
    main()
