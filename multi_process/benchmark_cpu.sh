#!/bin/bash

num_proc=1 	# number of processes expected to run concurrently
events=1 	# number of event each process will compute	
cores=1		# number of cores (sockets)
threads=1	# number of threads per core 
datapath=""	# data dir
while getopts n:e:c:t:p: flag;
do
    case "${flag}" in
        n) num_proc=${OPTARG};;
        e) events=${OPTARG};;
	c) cores=${OPTARG};;
	t) threads=${OPTARG};;
	p) datapath=${OPTARG};;
    esac
done
echo "$datapath"
echo "number of processes : $num_proc";
echo "number of events : $events";
export TRACCC_TEST_DATA_DIR=$datapath
Tstart=$(date "+%s.%3N")

#warm up run

../build/bin/traccc_seq_example --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_full/ttbar_mu200/ --events=$events --input-binary &
wait $!
result=$?
echo "result : $result" 

# end warm up run

for((i=0;i<num_proc;i++))
do
	# get processor id
	p=$((($i % ($cores * $threads))))
	echo " processor id $p";
	# end get processor id
	../build/bin/traccc_seq_example --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_full/ttbar_mu200/  --events=$events --input-binary &
done
wait
Tend=$(date "+%s.%3N")
elapsed=$(echo "scale=3; $Tend - $Tstart" | bc)
python3 log_data.py $num_proc $events $elapsed $cores $threads cpu
echo "Elapsed: $elapsed s"
exit $result
