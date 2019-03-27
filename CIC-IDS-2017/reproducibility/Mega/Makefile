
GOFLOWS=go-flows
PCAPS=/home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Monday-WorkingHours.pcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Wednesday-WorkingHours.pcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Thursday-WorkingHours.pcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap

all: mega.csv

AGM.csv: AGM.json ${PCAPS}
	${GOFLOWS} run features AGM.json export csv AGM.csv source libpcap ${PCAPS}
	
CAIA_Consensus.csv: CAIA_Consensus.json ${PCAPS}
	${GOFLOWS} run features CAIA_Consensus.json export csv CAIA_Consensus.csv source libpcap ${PCAPS}

TA.csv: TA_bidirectional.json ${PCAPS}
	${GOFLOWS} run features TA_bidirectional.json export csv TA.csv source libpcap ${PCAPS}

mega_unlabeled.csv: join.py AGM.csv CAIA_Consensus.csv TA.csv
	./join.py 
	
mega.csv: labeling.py mega_unlabeled.csv
	./labeling.py mega
	
CAIAConAGM_training.csv: train_test_split.py mega.csv
	./train_test_split.py mega

clean:
	rm -f mega_unlabeled.csv CAIA_Consensus.csv AGM.csv TA.csv

.SECONDARY: mega.csv mega_unlabeled.csv CAIA_Consensus.csv AGM.csv TA.csv

