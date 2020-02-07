#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

infile=sys.argv[1]
gtfile = sys.argv[2]

TIME_TOL = 60e3

# There are several entries in the ground truth file which share exactly the
# same start/end time and flow key. To avoid randomness during labeling,
# we here define which attack class to take in such cases. We try to give more
# specific attack descriptions a higher priority than more generic ones.
# Classes appearing first have higher priorities.
PRIORITIES = { attack: prio for prio, attack in enumerate([
	'Worms',
	'Backdoors',
	'Fuzzers',
	'Shellcode',
	'Exploits',
	'DoS',
	'Reconnaissance',
	'Analysis',
	'Generic',
	'Normal'
])}

print ('Loading ground truth.')
gt = pd.read_csv(gtfile)
del gt['Attack subcategory']
del gt['Attack Name']
del gt['Attack Reference']
del gt['.']
gt.rename(columns={'Start time': 'Stime', 'Last time': 'Ltime', 'Attack category': 'Attack'}, inplace=True)

print ('Adding join column.')

# For ICMP the port value contains several ICMP fields in some encoded
# format. Since this is not consistent with go-flows, ignore it.
mask = (gt.loc[:,'Protocol']=='icmp')
gt.loc[mask,['Source Port', 'Destination Port']] = 0

gt.loc[gt.loc[:,'Attack'].isna(), 'Attack'] = 'Normal'
gt.loc[:,'Attack'] = gt['Attack'].apply(lambda a: a.strip())
gt.loc[gt['Attack'] == 'Backdoor', 'Attack'] = 'Backdoors'
gt['attack_prio'] = gt['Attack'].apply(lambda a: PRIORITIES[a])

# Source and destination might be swapped in the ground truth, so
# define an ordering for the join operation

source = gt['Source Port'].map(str) + ' ' + gt['Source IP']
destination = gt['Destination Port'].map(str) + ' ' + gt['Destination IP']
mask = source > destination
gt['key'] = gt['Protocol'] + ' ' + source + ' ' + destination
gt.loc[mask, 'key'] = gt.loc[mask,'Protocol'] + ' ' + destination[mask] + ' ' + source[mask]
del source, destination

for k in ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']:
	del gt[k]

print ('Reading data.', end='', flush=True)
data = pd.read_csv('%s_unlabeled.csv' % infile)

print (' Shape:', data.shape)

print ('Adding join columns.', end='', flush=True)

modProto = data['protocolIdentifier'].apply(lambda n: { 1: 'icmp', 6: 'tcp', 17: 'udp'}[n])

modPort = data['destinationTransportPort'].copy()
mask = data['protocolIdentifier'] == 1 # ICMP
modPort[mask] = 0
source = data['sourceTransportPort'].map(str) + ' ' + data['sourceIPAddress']
destination = modPort.map(str) + ' ' + data['destinationIPAddress']
mask = source > destination
data['key'] = modProto + ' ' + source + ' ' + destination
data.loc[mask,'key'] = modProto[mask] + ' ' + destination[mask] + ' ' + source[mask]
del modProto, modPort, source, destination

data['flow_ind'] = data.index

print (' New shape:', data.shape)

print ('Merging with groundtruth.', end='', flush=True)
out = data.merge(gt, how='inner', on='key')

print (' New shape: ', out.shape)

print ('Dropping non-overlapping entries.', end='', flush=True)

out.query('(flowStartMilliseconds + flowDurationMilliseconds > Stime * 1000 - %d) & (flowStartMilliseconds < Ltime * 1000 + %d)' % (TIME_TOL, TIME_TOL), inplace=True)

print (' New shape:', out.shape)

print ('Dropping duplicate flows.', end='', flush=True)
out.sort_values(by=['flow_ind', 'attack_prio'], inplace=True) 
out.drop_duplicates(subset=['flow_ind'], inplace=True, keep='first')

print (' New shape:', out.shape)

print ('Dropping ground truth columns.', end='', flush=True)
for k in ['Stime', 'Ltime', 'attack_prio']:
	del out[k]
	
print (' New shape:', out.shape)

print ('Adding flows that do not exist in ground truth.', end='', flush=True)
data['Attack'] = 'Normal'
out = pd.concat( (out, data), sort=False, ignore_index=True)
out.drop_duplicates(subset=['flow_ind'], inplace=True, keep='first')
out.sort_values(by='flow_ind', inplace=True)

del out['flow_ind']
del out['key']

print (' New shape:', out.shape)

print ('Adding label.', end='', flush=True)
out['Label'] = (out['Attack'] != 'Normal').astype(int)

print (' Final shape:', out.shape)

print ('Writing.')

out.to_csv('%s.csv' % infile, index=False)


