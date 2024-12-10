# traccc extras

This directory is designed to hold miscellaneous bits and bobs that are
relevant to the traccc project. Examples include small scripts, write-ups,
small data files, and whatever else which may be useful to the traccc team, but
which does not fit in any of the other directories.

## Collecting throughput measurements
Collect throughput measurements using `traccc_throughput_mt_profiler.sh`

## Plotting throughput measurements

### Matplotlib based plotter
This python script easily does data preparation and plotting from raw output logfiles obtained through the throughput executables.
It does so using matplotlib. If you alternatively prefer to use ROOT for plotting, another script using PyROOT is available.

Usage: 
* Start by running a series of throughput measurements on a device or series of devices. Each device should log all its results in the same logfile. A separate script is available which runs through all available ttbar pile-up simulated events and a range of CPU threads and logs it. 
* Run `python traccc_throughput_mt_matplotter.py <datadir>/*.csv <outputdir>`

Note: If running on windows where wildcard expansion is not available, this should work instead:
`python traccc_throughput_mt_matplotter.py $(ls <datadir>\*.csv | % {$_.FullName + " "}) <outputdir>`

