# CIC-IDS-2017
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

### Usage from without the CN network
You can obtain a local copy of the dataset from [the dataset authors](https://www.unb.ca/cic/datasets/ids-2017.html). To reproduce the labeled csvs from a local copy, point the `DATASET` environment variable to the
directory containing the `*.pcap` files. For instance, `DATASET=. make CAIA` generates the labeled dataset for the CAIA feature vector if you download the `*.pcap` files to the directory where the `Makefile` is located.


# References
* I.  Sharafaldin,  A.  Habibi  Lashkari,  and  A.  A.  Ghorbani,  "Toward Generating a New Intrusion Detection Dataset and Intrusion  Traffic  Characterization,"  in ICISSP,  (Funchal,  Madeira,Portugal), pp. 108–116, SCITEPRESS, 2018.
* https://www.unb.ca/cic/datasets/ids-2017.html
