#################################################
#
#
# Name        : cisco_features.py
# Description : This script takes a json and outputs a CSV containing the CISCO analysis features
# Author      : Fares Meghdouri
#
# Notes : # not sure if I should create a set of supported cs also for the server
#            ( currently I am using the one extracted from client)    
#          # in the extensions, check if it exists or it is empty ? what is best ?
#
#################################################

import pandas as pd
import sys
import time

#################################################

client_offered_cs = set()
client_offered_cs_list = []
client_offered_extensions = set()
client_offered_extensions_list = []

#################################################


def get_cs(flow):
    global client_offered_cs
    try:
        for c in flow['cs']:
            client_offered_cs.add(c)
    except:
        pass
    return
# end def get_cs

def create_c_cs_vector(l_cs):
    global client_offered_cs_list
    vector = [0]*200#len(client_offered_cs_list)
    try:
        for cs in l_cs['cs']:
            vector[client_offered_cs_list.index(cs)] = 1
        return vector
    except:
        return vector
# end of def create_c_cs_vector

def create_s_cs_vector(l_cs):
    global client_offered_cs_list
    vector = [0]*200#len(client_offered_cs_list)
    try:
        vector[client_offered_cs_list.index(l_cs['scs'])] = 1
        return vector
    except:
        return vector
# end of def create_s_cs_vector

def get_cextensions(flow):
    global client_offered_extensions
    try:
        for e in flow['c_extensions']:
            client_offered_extensions.add(list(e.keys())[0])
    except:
        pass
    return
# end def get_cextensions

def create_c_extensions_vector(l_extensions):
    global client_offered_extensions_list
    vector = [0]*200#len(client_offered_extensions_list)
    try:
        for extension in l_extensions['c_extensions']:
            vector[client_offered_extensions_list.index(list(extension.keys())[0])] = 1
        return vector
    except:
        return vector
# end def create_c_extensions_vector

def create_s_extensions_vector(l_extensions):
    global client_offered_extensions_list
    vector = [0]*200#len(client_offered_extensions_list)
    try:
        for extension in l_extensions['s_extensions']:
            vector[client_offered_extensions_list.index(list(extension.keys())[0])] = 1
        return vector
    except:
        return vector
# end def create_s_extensions_vector

def create_features(name, length):
    """ returns a list of feature names using a counter ex: feat1, feat2, feat3... """
    features = []
    for i in range(int(length)):
        features.append(str(name) + "_" + str(i))
    return features
# end of def create_features

def PL_mapping(PL):
    """ map packet lengths to 10 markov states"""

    if PL < 150:
        return 0
    elif PL >= 150 and PL < 300:
        return 1
    elif PL >= 300 and PL < 450:
        return 2
    elif PL >= 450 and PL < 600:
        return 3
    elif PL >= 600 and PL < 750:
        return 4
    elif PL >= 750 and PL < 900:
        return 5
    elif PL >= 900 and PL < 1050:
        return 6
    elif PL >= 1050 and PL < 1200:
        return 7
    elif PL >= 1200 and PL < 1350:
        return 8
    elif PL >= 1350:
        return 9
# end def PL_mapping

def PT_mapping(PL):
    """ map inter-arrival-times to 10 markov states"""

    if PL < 50:
        return 0
    elif PL >= 50 and PL < 100:
        return 1
    elif PL >= 100 and PL < 150:
        return 2
    elif PL >= 150 and PL < 200:
        return 3
    elif PL >= 200 and PL < 250:
        return 4
    elif PL >= 250 and PL < 300:
        return 5
    elif PL >= 300 and PL < 350:
        return 6
    elif PL >= 350 and PL < 400:
        return 7
    elif PL >= 400 and PL < 450:
        return 8
    elif PL >= 450:
        return 9
# end def PT_mapping

def transition_vector(transitions):
    """ takes a list of transitions and returns
    a 100 long list of markov transision probabilities
    ps: change 'n' to number of states"""

    n = 10 #1+ max(transitions) #number of states
    M = [[0]*n for _ in range(n)]
    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1
    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return [item for sublist in M for item in sublist]
# end def transition_vector

def SPL(flow):
    """ takes packet lengths in a flow
    and returns a 100-long vector representing
    the markov transitions for 10 states"""
    try:
	    transitions = []
	    for packet in flow:
	        transitions.append(PL_mapping(packet['b']))
	    if len(transition_vector(transitions)) == 100:
	        return transition_vector(transitions)
	    else:
	        return [0]*100
    except:
        return [0]*100
#end def SPL

def SPT(flow):
    """ takes inter-arrival-times in a flow
    and returns a 100-long vector representing
    the markov transitions for 10 states"""
    try:
	    transitions = []
	    for packet in flow:
	        transitions.append(PT_mapping(packet['ipt']))
	    if len(transition_vector(transitions)) == 100:
	        return transition_vector(transitions)
	    else:
	        return [0]*100
    except:
        return [0]*100
#end def SPT

def get_pubkey_length(flow):
    try:
        return int(flow["c_key_length"])
    except:
        return 0
# end def get_pubkey_length

def get_s_num_certificates(flow):
    try:
        return len(flow['s_cert'])
    except:
        return 0
# end def get_s_num_certificates

def get_s_num_san(flow):
    count = 0
    try:
        for certificate in flow['s_cert']:
            for extension in certificate['extensions']:
                try:
                    local_count = len(extension['X509v3 Subject Alternative Name'].split(','))
                    count += local_count
                except:
                    continue
        return count
    except:
        return 0
# end def get_s_num_san

def main():
    global client_offered_cs_list, client_offered_extensions_list
    # read the data
    print(">> Reading the data : file > {}".format(input_name))
    dataframe = pd.read_json(input_name)

    # save some memory
    dataframe.drop(['byte_dist_mean',
                    'byte_dist_std',
                    'bytes_in',
                    'bytes_out',
                    'debug',
                    'num_pkts_in',
                    'num_pkts_out',
                    'probable_os',
                    'time_end'],
                     axis=1)

    # create a final dataframe that will be exported
    output = dataframe[['sa','da','sp','dp','pr','time_start']].copy()

    ###################   SPLT   ###################
    print('>> Get the SPLT features')
    # get markov transitions for packets lengths
    output['SPL'] = dataframe['packets'].apply(SPL)

    # get markov transitions for inter-arrival-times
    output['SPT'] = dataframe['packets'].apply(SPT)

    ###################    BD    ###################
    # get packet lengths distribution
    output['byte_dist'] = dataframe['byte_dist']

    # save memory
    dataframe.drop(['byte_dist',
                    'packets'],
                     axis=1)

    ###################    TLS    ###################
    print('>> Get the TLS features')
    ### get Client data
    # get client ciphersuites
    dataframe['tls'].apply(get_cs)
    client_offered_cs_list = list(client_offered_cs)
    output['c_cs'] = dataframe['tls'].apply(create_c_cs_vector)

    # get client extensions
    dataframe['tls'].apply(get_cextensions)
    client_offered_extensions_list = list(client_offered_extensions)
    output['c_extensions'] = dataframe['tls'].apply(create_c_extensions_vector)

    # get client public key length
    output['c_pubkey_length'] = dataframe['tls'].apply(get_pubkey_length)

    ### get server data
    # get server selected ciphersuite
    output['s_cs'] = dataframe['tls'].apply(create_s_cs_vector)

    # get server extensions
    output['s_extensions'] = dataframe['tls'].apply(create_s_extensions_vector)

    # get numbert of server certificates
    output['s_num_cert'] = dataframe['tls'].apply(get_s_num_certificates)

    # get number of SAN names
    output['s_num_SAN'] = dataframe['tls'].apply(get_s_num_san)

    # get validity in days
    # TODO: implement this | problem : # many certificates -> many validity intervals
    #                                   # some intervals are wrong and there is no logic behind them

    # self-signed certificate check
    # TODO: implement this | problem : # which field to read for owner? (issuer is already there)

    dataframe.drop(['tls'], axis=1)

    ###################    DNS    ###################
    print('>> Get the DNS features')
    # TODO implement this | problem : # no data example available to know which inputs are there

    ###################    HTTP   ###################
    print('>> Get the HTTP features')
    # TODO: implement this | problem : # no data example available to know which inputs are there

    #################################################
    ########## fixing the final dataframe ###########

    print('>> Cleaning')
    del dataframe

    print(">> Mapping into dummy features")
    output[create_features('SPL', 100)] = pd.DataFrame(output.SPL.values.tolist(), index= output.index)
    output = output.drop(['SPL'], axis=1)

    output[create_features('SPT', 100)] = pd.DataFrame(output.SPT.values.tolist(), index= output.index)
    output = output.drop(['SPT'], axis=1)

    output[create_features('byte_dist', 256)] = pd.DataFrame(output.byte_dist.values.tolist(), index= output.index)
    output = output.drop(['byte_dist'], axis=1)

    output[create_features('c_cs', 200)] = pd.DataFrame(output.c_cs.values.tolist(), index= output.index)
    output = output.drop(['c_cs'], axis=1)

    output[create_features('c_extensions', 200)] = pd.DataFrame(output.c_extensions.values.tolist(), index= output.index)
    output = output.drop(['c_extensions'], axis=1)

    output[create_features('s_cs', 200)] = pd.DataFrame(output.s_cs.tolist(), index= output.index)
    output = output.drop(['s_cs'], axis=1)

    output[create_features('s_extensions', 200)] = pd.DataFrame(output.s_extensions.values.tolist(), index= output.index)
    output = output.drop(['s_extensions'], axis=1)


    #################################################
    output = output.loc[:, (output != 0).any(axis=0)]
    #################################################
    ########## saving the final dataframe ###########
    print(">> Saving : file > {}".format(output_name))
    output.to_csv(output_name, index = False)

    print(">> Done")
# end def main

if __name__ == "__main__":
    input_name = sys.argv[1]
    output_name = sys.argv[2]
    main()
