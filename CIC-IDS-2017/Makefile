VECTORS:=CAIA Consensus TA AGM_10s AGM_60s AGM_3600s CAIAConAGM Mega1 Mega2

GOFLOWS?=go-flows
PYTHON?=python3
DATASET?=.

PCAPS:=Monday-WorkingHours.pcap Tuesday-WorkingHours.pcap Wednesday-WorkingHours.pcap Thursday-WorkingHours.pcap Friday-WorkingHours.pcap
PCAPFILES=$(addprefix ${DATASET}/, ${PCAPS})

SSHARGS:=-o ControlMaster=auto -o ControlPath=./.sshcontrol -o ControlPersist=10s

# allow variables to be set/overridden
-include makeenv

# set a few go-flows parameters explicitly to be on the safe side considering changing defaults
GFARGS:=-sort start


all: ${VECTORS}


# rules for getting pcaps
.sshcontrol:
	read -p 'User for fileserver.cn.tuwien.ac.at: ' SSHUSER && ssh ${SSHARGS} $$SSHUSER@fileserver.cn.tuwien.ac.at :

%.pcap: .sshcontrol
	scp ${SSHARGS} fileserver.cn.tuwien.ac.at:/home/cn_group/datasets/CIC-IDS-2017/PCAPs/$*.pcap .



# rules for flow extraction
%_unlabeled.csv: flow_specifications/%.json ${PCAPFILES}
	${GOFLOWS} run ${GFARGS} features flow_specifications/$*.json export csv $*_unlabeled.csv source libpcap ${PCAPFILES}

Mega1_unlabeled.csv: flow_specifications/CAIA_Consensus_AGM_TA.json ${PCAPFILES}
	${GOFLOWS} run ${GFARGS} features flow_specifications/CAIA_Consensus_AGM_TA.json export csv Mega1_unlabeled.csv source libpcap ${PCAPFILES}



# rules for joining
CAIAConAGM_unlabeled.csv: extra_preprocessing/join_CAIAConAGM.py CAIA_Consensus_unlabeled.csv AGM_10s_unlabeled.csv
	${PYTHON} extra_preprocessing/join_CAIAConAGM.py

Mega2_unlabeled.csv: extra_preprocessing/join_Mega.py CAIA_Consensus_unlabeled.csv AGM_10s_unlabeled.csv TA_unlabeled.csv
	${PYTHON} extra_preprocessing/join_Mega.py



# rules for labeling
%.csv: labeling/labeling.py %_unlabeled.csv
	${PYTHON} labeling/labeling.py $* 5tuple

AGM_%.csv: labeling/labeling.py AGM_%_unlabeled.csv
	${PYTHON} labeling/labeling.py AGM_$* AGM

TA.csv: labeling/labeling.py TA_unlabeled.csv
	${PYTHON} labeling/labeling.py TA 2tuple


# misc
${VECTORS}: %: %.csv

clean:
	rm -fv *_unlabeled.csv

.PHONY: ${VECTORS} all clean

# allow intermediate files to be deleted
.SECONDARY:

