import pandas as pd
import numpy as np
import datetime

data = pd.DataFrame()

def read_data(day):
	global data
	print(">> Labeling {}".format(day))
	print(">> Loading the data")
	data = pd.read_csv('{}.csv'.format(day))
	data['Attack'] = 'Normal'
	data['Label']  = '0'

def save_data(day):
	global data
	print(">> Saving {}".format(day))
	data.to_csv("{}_labeled.csv".format(day), index= False)
	del data
	print("#"*20)

def time_fixing(day, hours, minutes):
	return (datetime.datetime(2017, 7, day, hours + 3, minutes) - datetime.datetime(1970,1,1)).total_seconds()

def setting_label(sip="", dip="", sport="", dport="", st="", et="", attack=""):
	
	print(">> Adding {}".format(attack))

	if sip and dip and dport:
		data['Attack'] = np.where((data['sourceIPAddress'] == sip) & 
								  (data['destinationIPAddress'] == dip) &
								  (str(data['destinationTransportPort']) == dport) &
								  (data['flowStartSeconds'] > st) &
								  (data['flowStartSeconds'] < et),
								   attack, data['Attack'])
		return

	if sip and dip:
		data['Attack'] = np.where((data['sourceIPAddress'] == sip) & 
								  (data['destinationIPAddress'] == dip) &
								  (data['flowStartSeconds'] > st) &
								  (data['flowStartSeconds'] < et),
								   attack, data['Attack'])
		return

	if sip:
		data['Attack'] = np.where((data['sourceIPAddress'] == sip) &
								  (data['flowStartSeconds'] > st) &
								  (data['flowStartSeconds'] < et),
								   attack, data['Attack'])
		return

	if dip:
		data['Attack'] = np.where((data['destinationIPAddress'] == dip) &
								  (data['flowStartSeconds'] > st) &
								  (data['flowStartSeconds'] < et),
								   attack, data['Attack'])
		return

# here start ######################################################################

print(">> Welcome to the labeling script <<")
###################################################################
read_data("Monday")
save_data("Monday")
###################################################################
read_data("Tuesday")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(4, 9, 18), et=time_fixing(4, 9, 22),
	          attack="Brute Force:FTP-Patator")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(4, 13, 58), et=time_fixing(4, 15, 2),
	          attack="Brute Force:SSH-Patator")

data['Label'] = np.where(data['Attack'] == "Normal", 0, 1)

save_data("Tuesday")
###################################################################
read_data("Wednesday")

setting_label(sip="172.16.0.1", dip="192.168.10.51", dport="444",
	          st=time_fixing(5, 15, 10), et=time_fixing(5, 15, 34),
	          attack="DoS / DDoS:Heartbleed")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(5, 9, 45), et=time_fixing(5, 10, 12),
	          attack="DoS / DDoS:DoS slowloris")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(5, 10, 12), et=time_fixing(5, 10, 37),
	          attack="DoS / DDoS:DoS Slowhttptest")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(5, 10, 41), et=time_fixing(5, 11, 2),
	          attack="DoS / DDoS:DoS Hulk")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(5, 11, 8), et=time_fixing(5, 11, 25),
	          attack="DoS / DDoS:DoS GoldenEye")

data['Label'] = np.where(data['Attack'] == "Normal", 0, 1)

save_data("Wednesday")
###################################################################
read_data("Thursday")

setting_label(sip="172.16.0.1", dip="192.168.10.51",
	          st=time_fixing(6, 9, 18), et=time_fixing(6, 10, 2),
	          attack="Web Attack:Brute Force")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(6, 10, 13), et=time_fixing(6, 10, 37),
	          attack="Web Attack:XSS")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(6, 10, 38), et=time_fixing(6, 10, 44),
	          attack="Web Attack:Sql Injection")

setting_label(sip="172.16.0.1", dip="192.168.10.8", 
	          st=time_fixing(6, 14, 17), et=time_fixing(6, 14, 37),
	          attack="Infiltration:Dropbox download")

setting_label(sip="205.174.165.73", dip="192.168.10.25", 
	          st=time_fixing(6, 14, 51), et=time_fixing(6, 15, 5),
 	          attack="Infiltration:Cool disk")

setting_label(sip="192.168.10.8",
	          st=time_fixing(6, 15, 2), et=time_fixing(6, 15, 47),
	          attack="Infiltration:Dropbox download - (Portscan + Nmap) from victim")

data['Label'] = np.where(data['Attack'] == "Normal", 0, 1)

save_data("Thursday")
###################################################################
read_data("Friday")

setting_label(sip="205.174.165.73",
	          st=time_fixing(7, 10, 0), et=time_fixing(7, 11, 4),
	          attack="Botnet:ARES")

setting_label(dip="205.174.165.73", 
	          st=time_fixing(7, 10, 0), et=time_fixing(7, 11, 4),
	          attack="Botnet:ARES")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(7, 13, 53), et=time_fixing(7, 14, 37),
	          attack="PortScan:PortScan - Firewall on")

setting_label(sip="205.174.165.73", dip="192.168.10.50", 
	          st=time_fixing(7, 14, 49), et=time_fixing(7, 15, 31),
	          attack="PortScan:PortScan - Firewall off")

setting_label(sip="172.16.0.1", dip="192.168.10.50", 
	          st=time_fixing(7, 15, 54), et=time_fixing(6, 16, 18),
	          attack="DDoS:LOIT")

data['Label'] = np.where(data['Attack'] == "Normal", 0, 1)

save_data("Friday")
###################################################################