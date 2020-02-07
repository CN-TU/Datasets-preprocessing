VECTORS:=CAIA Consensus TA AGM_10s AGM_60s AGM_3600s CAIAConAGM Mega1 Mega2 packet

GOFLOWS?=go-flows
PYTHON?=python3
DATASET?=

PCAPS2201:=$(addprefix "UNSW-NB15 Source-Files"/"UNSW-NB15 - pcap files"/"pcaps 22-1-2015"/, $(addsuffix .pcap, $(shell seq 53)))
PCAPS1702:=$(addprefix "UNSW-NB15 Source-Files"/"UNSW-NB15 - pcap files"/"pcaps 17-2-2015"/, $(addsuffix .pcap, $(shell seq 27)))
FILELIST:=${PCAPS2201} ${PCAPS1702}


# allow variables to be set/overridden
-include makeenv

# set a few go-flows parameters explicitly to be on the safe side considering changing defaults
GFARGS:=-sort start

GF_CMD:=${PYTHON} extra_preprocessing/gfstream.py ${GOFLOWS}

# from within TU we directly load any data from the fileserver, otherwise
# require a local copy specified by ${DATASET}
ifeq ($(strip ${DATASET}),)

SSHARGS:=-o Ciphers=aes128-ctr -o ControlMaster=auto -o ControlPath=./.sshcontrol -o ControlPersist=10s
UNZIP_CMD=ssh ${SSHARGS} fileserver.cn.tuwien.ac.at unzip -p /home/cn_group/datasets/UNSW-NB15/UNSW-NB15_Source-files.zip
GROUNDTRUTH=UNSW-NB15_GT.csv

.sshcontrol:
	read -p 'User for fileserver.cn.tuwien.ac.at: ' SSHUSER && ssh ${SSHARGS} $$SSHUSER@fileserver.cn.tuwien.ac.at :

UNSW-NB15_GT.csv: .sshcontrol
	scp ${SSHARGS} fileserver.cn.tuwien.ac.at:/home/cn_group/datasets/UNSW-NB15/UNSW-NB15_GT.csv .
	
else

UNZIP_CMD=unzip -p ${DATASET}/UNSW-NB15_Source-files.zip
GROUNDTRUTH=${DATASET}/UNSW-NB15_GT.csv

endif

# used by gfstream.py
export UNZIP_CMD 


all: ${VECTORS}


# rules for flow extraction
%_unlabeled.csv: extra_preprocessing/gfstream.py flow_specifications/%.json .sshcontrol
	${GF_CMD} run ${GFARGS} features flow_specifications/$*.json export csv $*_unlabeled.csv source libpcap -filter inbound ${FILELIST}

Mega1_unlabeled.csv: extra_preprocessing/gfstream.py flow_specifications/CAIA_Consensus_AGM_TA.json .sshcontrol
	${GF_CMD} run ${GFARGS} features flow_specifications/CAIA_Consensus_AGM_TA.json export csv Mega1_unlabeled.csv source libpcap -filter inbound ${FILELIST}


# rules for joining
CAIAConAGM_unlabeled.csv: extra_preprocessing/join_CAIAConAGM.py CAIA_Consensus_unlabeled.csv AGM_10s_unlabeled.csv
	${PYTHON} extra_preprocessing/join_CAIAConAGM.py

Mega2_unlabeled.csv: extra_preprocessing/join_Mega.py CAIA_Consensus_unlabeled.csv AGM_10s_unlabeled.csv TA_unlabeled.csv
	${PYTHON} extra_preprocessing/join_Mega.py


# rules for labeling
%.csv: ${GROUNDTRUTH} labeling/labeling_5tuple.py %_unlabeled.csv
	${PYTHON} labeling/labeling_5tuple.py $* ${GROUNDTRUTH}


# misc
${VECTORS}: %: %.csv

clean:
	rm -fv *_unlabeled.csv

.PHONY: ${VECTORS} all clean

# delete target if a recipe fails
.DELETE_ON_ERROR:

# allow intermediate files to be deleted
.SECONDARY:

