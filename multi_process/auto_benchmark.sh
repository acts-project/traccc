#!/bin/bash

# run n cuda processes for example n in range 1 -> 32
# benchmark cuda mps for increments of 10 events per process for example i in range 1 -> 50
max_proc=16	# number of processes upper bound
max_events=150	# maximum number of events per process
increment=10 	# event steps 
cores=1		# number of physical cores
threads=1	# number of threads per physical core 
gpu=1		# number of gpus
path='../data'	# data path
run_cpu=0	# 1-cpu and 0-cuda 

while getopts r:c:t:p:e:g:d: flag;
do
    case "${flag}" in
	c) cores=${OPTARG};;
	t) threads=${OPTARG};;
	p) max_proc=${OPTARG};;
	e) max_events=${OPTARG};;
	g) gpu=${OPTARG};;
	d) path=${OPTARG};;
	r) run_cpu=${OPTARG};;
    esac
done

echo "$max_proc $max_events";
for((i=max_events;i<=max_events;i+=increment))
do	
	echo "starting to benchmark with $i processes";
	for((j=1;j<=max_proc;j+=1))
	do
		echo "starting new run with $j events";
		if [ $run_cpu != 0 ];then
				
			./benchmark_cpu.sh -p"$path" -n$j -e$i -c$cores -t$threads	
			result=$?
		else
			./benchmark_cuda.sh -p"$path" -n$j -e$i -c$cores -t$threads -g$gpu
			result=$?
		fi
		if [ $result != 0 ];then
			exit $result
		fi		
		sleep 1
	done		
done
echo quit|nvidia-cuda-mps-control

exit $result
