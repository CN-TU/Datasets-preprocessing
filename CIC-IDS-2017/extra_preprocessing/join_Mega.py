#!/usr/bin/env python3

import pandas as pd

pd.merge_asof (
	pd.merge_asof (
		pd.read_csv('CAIA_Consensus.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
		pd.read_csv('AGM.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
		on='flowStartMilliseconds',
		direction='backward',
		by='sourceIPAddress'
		),
	pd.read_csv('TA.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
	on='flowStartMilliseconds',
	direction='backward',
	by=['sourceIPAddress','destinationIPAddress']
).to_csv('mega_unlabeled.csv', index=False)
