# UNSW-NB15

## Automated processing using the `Makefile`
### Requirements
* Python 3
* A compiled binary of [go-flows](https://github.com/CN-TU/go-flows)
* Functionality has only been verified with GNU Make under Linux

### Usage
Run `make <feature vector>` to build a labeled csv, where available feature vectors can be found in the first line of the [`Makefile`](Makefile).
For instance, `make CAIA` produces a `CAIA.csv` with preprocessed, labeled flows.

If the [go-flows](https://github.com/CN-TU/go-flows) binary is not contained in the path, its location can be provided by
the `GOFLOWS` environment variable. Environment variables can alternatively be specified in a file called `makeenv`.

### Usage from outside the CN network
You can obtain a local copy of the dataset from [the dataset authors](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/). To reproduce the labeled csvs from a local copy, point the `DATASET` environment variable to the
directory containing the file `UNSW-NB15_Source-files.zip`. For instance, `DATASET=. make CAIA` generates the labeled dataset for the CAIA feature vector if you download the file to the directory where the `Makefile` is located.

# Known dataset issues
* The ground truth file contains several duplicates entries and matching with flows from the provided pcaps is difficult.
* The majority of packets occur twice in the dataset, presumably because capturing was performed on both the incoming and outgoing interfaces of a router.
* The last pcap file `pcaps 17-2-2015/27.pcap` is truncated.

# References
* N.  Moustafa  and  J.  Slay,  "UNSW-NB15:  a  comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," in MilCIS, pp. 1–6, Nov. 2015.
* https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
