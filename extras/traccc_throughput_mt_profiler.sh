#!/bin/bash
#
# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0
#
# Simple script running the selected instance of the multi-threaded throughput
# executable on a whole set of "measurement points".
#

# Stop on errors.
set -e

# Function printing the usage information for the script.
usage() {
   echo "Script running a suite of multi-threaded throughput tests"
   echo ""
   echo "Usage: traccc_throughput_mt_profiling.sh [options]"
   echo ""
   echo "Basic options:"
   echo "  -x <executable>      Selects the executable to use"
   echo "  -t <maxThreads>      Maximum number of threads to test"
   echo "  -r <repetitions>     The number of repetitions in the test"
   echo "  -e <eventMultiplier> Multiplier for the number of events per thread"
   echo "  -c <csvFile>         Name of the output CSV file"
   echo "  -h                   Print this help"
   echo ""
   echo "Detailed options:"
   echo "  -g <detectorGeomFile> Detector geometry CSV file"
   echo "  -d <digiConfigFile>   Detector digitization JSON file"
   echo ""
}

# Parse the command line arguments.
TRACCC_EXECUTABLE=${TRACCC_EXECUTABLE:-"traccc_throughput_mt"}
TRACCC_MAX_THREADS=${TRACCC_MAX_THREADS:-$(nproc)}
TRACCC_REPETITIONS=${TRACCC_REPETITIONS:-1}
TRACCC_EVT_MULTI=${TRACCC_EVT_MULTI:-"1"}
TRACCC_CSV_FILE=${TRACCC_CSV_FILE:-"output.csv"}
TRACCC_DET_FILE=${TRACCC_DET_FILE:-"tml_detector/trackml-detector.csv"}
TRACCC_DIGI_FILE=${TRACCC_DIGI_FILE:-"tml_detector/default-geometric-config-generic.json"}
while getopts ":x:t:r:e:c:g:d:h" opt; do
   case $opt in
      x)
         TRACCC_EXECUTABLE=$OPTARG
         ;;
      t)
         TRACCC_MAX_THREADS=$OPTARG
         ;;
      r)
         TRACCC_REPETITIONS=$OPTARG
         ;;
      e)
         TRACCC_EVT_MULTI=$OPTARG
         ;;
      c)
         TRACCC_CSV_FILE=$OPTARG
         ;;
      g)
         TRACCC_DET_FILE=$OPTARG
         ;;
      d)
         TRACCC_DIGI_FILE=$OPTARG
         ;;
      h)
         usage
         exit 0
         ;;
      :)
         echo "Argument -$OPTARG requires a parameter!"
         usage
         exit 1
         ;;
      ?)
         echo "Unknown argument: -$OPTARG"
         usage
         exit 1
         ;;
   esac
done

# Print the configuration received.
echo "Using configuration:"
echo "   EXECUTABLE  : ${TRACCC_EXECUTABLE}"
echo "   MAX_THREADS : ${TRACCC_MAX_THREADS}"
echo "   REPETITIONS : ${TRACCC_REPETITIONS}"
echo "   EVT_MULTI   : ${TRACCC_EVT_MULTI}"
echo "   CSV_FILE    : ${TRACCC_CSV_FILE}"
echo "   DET_FILE    : ${TRACCC_DET_FILE}"
echo "   DIGI_FILE   : ${TRACCC_DIGI_FILE}"

# Check whether the output file already exists. Refuse to overwrite existing
# files.
if [[ -f "${TRACCC_CSV_FILE}" ]]; then
   echo "***"
   echo "*** Will not overwrite ${TRACCC_CSV_FILE}!"
   echo "***"
   exit 1
fi

# The input directories to use.
TRACCC_INPUT_DIRS=("ttbar_mu20"  "ttbar_mu40"  "ttbar_mu60"  "ttbar_mu80"
                   "ttbar_mu100" "ttbar_mu140" "ttbar_mu200" "ttbar_mu300")

# The number of events to process for the different mu values. Chosen to take
# roughly the same amount of time to process on a CPU.
declare -A TRACCC_EVT_COUNT
TRACCC_EVT_COUNT["ttbar_mu20"]=$((1000*${TRACCC_EVT_MULTI}))
TRACCC_EVT_COUNT["ttbar_mu40"]=$((500*${TRACCC_EVT_MULTI}))
TRACCC_EVT_COUNT["ttbar_mu60"]=$((250*${TRACCC_EVT_MULTI}))
TRACCC_EVT_COUNT["ttbar_mu80"]=$((200*${TRACCC_EVT_MULTI}))
TRACCC_EVT_COUNT["ttbar_mu100"]=$((150*${TRACCC_EVT_MULTI}))
TRACCC_EVT_COUNT["ttbar_mu140"]=$((90*${TRACCC_EVT_MULTI}))
TRACCC_EVT_COUNT["ttbar_mu200"]=$((50*${TRACCC_EVT_MULTI}))
TRACCC_EVT_COUNT["ttbar_mu300"]=$((20*${TRACCC_EVT_MULTI}))

# Put a header on the CSV file.
echo "directory,threads,loaded_events,cold_run_events,processed_events,target_cells_per_partition,warm_up_time,processing_time" \
   > "${TRACCC_CSV_FILE}"

# Counter for a nice printout.
COUNTER=1
COUNT=$((${#TRACCC_INPUT_DIRS[@]}*${TRACCC_MAX_THREADS}*${TRACCC_REPETITIONS}))

# Iterate over the number of threads.
for NTHREAD in $(seq 1 ${TRACCC_MAX_THREADS}); do
   # Iterate over the input datasets.
   for EVTDIR in ${TRACCC_INPUT_DIRS[@]}; do
      # Perform the requested number of repetitions.
      for REPEAT in $(seq 1 ${TRACCC_REPETITIONS}); do

         # Tell the user what's happening.
         echo ""
         echo "Running test ${COUNTER} / ${COUNT}"
         ((COUNTER++))

         # Run the throughput test.
         ${TRACCC_EXECUTABLE}                                                 \
            --detector_file="${TRACCC_DET_FILE}"                              \
            --digitization_config_file="${TRACCC_DIGI_FILE}"                  \
            --input_directory="tml_full/${EVTDIR}/"                           \
            --threads=${NTHREAD}                                              \
            --cold_run_events=$((20*${NTHREAD}))                              \
            --processed_events=$((${TRACCC_EVT_COUNT[${EVTDIR}]}*${NTHREAD})) \
            --log_file="${TRACCC_CSV_FILE}"
      done
   done
done
