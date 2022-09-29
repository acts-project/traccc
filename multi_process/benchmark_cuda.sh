#!/bin/bash

num_proc=1 	# number of processes expected to run concurrently
events=1 	# number of event each process will compute	
cores=1		# number of cores (sockets)
threads=1	# number of threads per core 
datapath=""	# data dir
numgpus=1	# number of gpus

while getopts n:e:c:t:p:g: flag;
do
    case "${flag}" in
        n) num_proc=${OPTARG};;
        e) events=${OPTARG};;
	c) cores=${OPTARG};;
	t) threads=${OPTARG};;
	p) datapath=${OPTARG};;
	g) numgpus=${OPTARG};;
    esac
done
echo "$datapath"
echo "number of processes : $num_proc";
echo "number of events : $events";
# echo "log path $log_dir"
export TRACCC_TEST_DATA_DIR=$datapath

# enable cuda mps
nvidia-cuda-mps-control -d
mps_ret=$?
if [ $mps_ret -ne 0 ]; then
    echo "Failed to enabled CUDA mps : $mps_ret"
    exit $mps_ret
fi
echo "Enabled CUDA mps"
# end enable cuda mps

# warmup / test run
CUDA_VISIBLE_DEVICES=0 ../build/bin/traccc_seq_example_cuda --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_full/ttbar_mu200/ --events=$events --input-binary &
wait $!
result=$?
echo "result : $result" 
# end warm up/ test run

Tstart=$(date "+%s.%3N")
for((i=0;i<num_proc;i++))
do
	p=$((($i % ($cores * $threads))))
	echo " processor id $p";
	# get gpu id
	gpu_id=$(($i % $numgpus))
	echo " gpu $gpu_id";
	# end get gpu id

	# start job
	CUDA_VISIBLE_DEVICES=$gpu_id ../build/bin/traccc_seq_example_cuda --detector_file=tml_detector/trackml-detector.csv --digitization_config_file=tml_detector/default-geometric-config-generic.json --input_directory=tml_full/ttbar_mu200/  --events=$events --input-binary &
done
wait
Tend=$(date "+%s.%3N")
elapsed=$(echo "scale=3; $Tend - $Tstart" | bc)
python3 log_data.py $num_proc $events $elapsed $cores $threads cuda
echo "Elapsed: $elapsed s"
exit $result
