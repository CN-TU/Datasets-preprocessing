# UNSW-NB15
## Usage with a local dataset copy
To reproduce the labeled csvs from a local copy of the dataset, point the `DATASET` environment variable to the directory containing `UNSW-NB15_Source-files.zip` and `UNSW-NB15_GT.csv`, and call e.g. `make CAIA` to generate the labeled dataset for the CAIA feature vector.

If the [go-flows](https://github.com/CN-TU/go-flows) binary is not contained in the path, its location can be provided by the `GOFLOWS` environment variable. Environment variables can alternatively specified by a file called `makeenv`.

## Known dataset issues
* The ground truth file contains several duplicates entries and matching with flows from the provided pcaps is difficult.
* The majority of packets occur twice in the dataset, presumably because capturing was performed on both the incoming and outgoing interfaces of a router.
* The last pcap file `pcaps 17-2-2015/27.pcap` is truncated.
