#!/usr/bin/env python3

import pandas as pd

pd.merge_asof (
	pd.merge_asof (
		pd.read_csv('CAIA_Consensus_unlabeled.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
		pd.read_csv('AGM_10s_unlabeled.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
		on='flowStartMilliseconds',
		direction='backward',
		by='sourceIPAddress'
		),
	pd.read_csv('TA_unlabeled.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
	on='flowStartMilliseconds',
	direction='backward',
	by=['sourceIPAddress','destinationIPAddress']
).to_csv('Mega2_unlabeled.csv', index=False)
