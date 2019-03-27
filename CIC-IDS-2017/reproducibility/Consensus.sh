echo "expoting"
go-flows run -sort start features Consensus.json export csv Monday.csv source libpcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Monday-WorkingHours.pcap
go-flows run -sort start features Consensus.json export csv Tuesday.csv source libpcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap
go-flows run -sort start features Consensus.json export csv Wednesday.csv source libpcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Wednesday-WorkingHours.pcap
go-flows run -sort start features Consensus.json export csv Thursday.csv source libpcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Thursday-WorkingHours.pcap
go-flows run -sort start features Consensus.json export csv Friday.csv source libpcap /home/meghdouri/Desktop/datasets/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap

#echo "uni to bidi"
#python bidi.py

echo "labeling"
python labeling_rules.py
echo "concatenating"
sed '1d' Tuesday_labeled.csv > tmp1.csv
sed '1d' Wednesday_labeled.csv > tmp2.csv
sed '1d' Thursday_labeled.csv > tmp3.csv
sed '1d' Friday_labeled.csv > tmp4.csv

cat Monday_labeled.csv tmp1.csv tmp2.csv tmp3.csv tmp4.csv > Full_Consensus.csv

echo "cleaning"
rm tmp1.csv
rm tmp2.csv
rm tmp3.csv
rm tmp4.csv
rm Monday_labeled.csv
rm Tuesday_labeled.csv
rm Wednesday_labeled.csv
rm Thursday_labeled.csv
rm Friday_labeled.csv
rm Monday.csv
rm Tuesday.csv
rm Wednesday.csv
rm Thursday.csv
rm Friday.csv
rm Monday_bidi.csv
rm Tuesday_bidi.csv
rm Wednesday_bidi.csv
rm Thursday_bidi.csv
rm Friday_bidi.csv

python freq_tables.py ohe Full_Consensus.csv 20 98 test.csv

python train_test_split.py

echo "done"
