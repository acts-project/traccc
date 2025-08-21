#!/bin/bash
#
# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2023-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0
#
# Simple script running the selected instance of the multi-threaded throughput
# executable on a whole set of ITk ttbar simulations, with different pileup
# values.
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
   echo "  -i <inputDir>        Selects the input directory with all files"
   echo "  -m <minThreads>      Minimum number of threads to test"
   echo "  -t <maxThreads>      Maximum number of threads to test"
   echo "  -s <threadStep>      Steps to increase the thread count by"
   echo "  -r <repetitions>     The number of repetitions in the test"
   echo "  -e <eventMultiplier> Multiplier for the number of events per thread"
   echo "  -c <csvFile>         Name of the output CSV file"
   echo "  -y <throughputType>  Type of throughput test to run (traccc/g200/g100)"
   echo "  -h                   Print this help"
   echo ""
}

# Parse the command line arguments.
TRACCC_EXECUTABLE=${TRACCC_EXECUTABLE:-"traccc_throughput_mt"}
TRACCC_INPUT_DIR=${TRACCC_INPUT_DIR:-"ATLAS-P2-RUN4-03-00-01/"}
TRACCC_MIN_THREADS=${TRACCC_MIN_THREADS:-1}
TRACCC_MAX_THREADS=${TRACCC_MAX_THREADS:-$(nproc)}
TRACCC_THREAD_STEP=${TRACCC_THREAD_STEP:-1}
TRACCC_REPETITIONS=${TRACCC_REPETITIONS:-5}
TRACCC_CSV_FILE=${TRACCC_CSV_FILE:-"output.csv"}
TRACCC_THROUGPUT_TYPE=${TRACCC_THROUGPUT_TYPE:-"traccc"}
while getopts ":x:i:m:t:r:c:y:h" opt; do
   case $opt in
      x)
         TRACCC_EXECUTABLE=$OPTARG
         ;;
      i)
         TRACCC_INPUT_DIR=$OPTARG
         ;;
      m)
         TRACCC_MIN_THREADS=$OPTARG
         ;;
      t)
         TRACCC_MAX_THREADS=$OPTARG
         ;;
      s)
         TRACCC_THREAD_STEP=$OPTARG
         ;;
      r)
         TRACCC_REPETITIONS=$OPTARG
         ;;
      c)
         TRACCC_CSV_FILE=$OPTARG
         ;;
      y)
         TRACCC_THROUGPUT_TYPE=$OPTARG
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
echo "   EXECUTABLE      : ${TRACCC_EXECUTABLE}"
echo "   INPUT_DIR       : ${TRACCC_INPUT_DIR}"
echo "   MIN_THREADS     : ${TRACCC_MIN_THREADS}"
echo "   MAX_THREADS     : ${TRACCC_MAX_THREADS}"
echo "   THREAD_STEP     : ${TRACCC_THREAD_STEP}"
echo "   REPETITIONS     : ${TRACCC_REPETITIONS}"
echo "   CSV_FILE        : ${TRACCC_CSV_FILE}"
echo "   THROUGHPUT_TYPE : ${TRACCC_THROUGPUT_TYPE}"

# Check whether the output file already exists. Refuse to overwrite existing
# files.
if [[ -f "${TRACCC_CSV_FILE}" ]]; then
   echo "***"
   echo "*** Will not overwrite ${TRACCC_CSV_FILE}!"
   echo "***"
   exit 1
fi

# Additional flags tuning the cuts for the G100/G200 pipelines.
G200_CUTS=(--seedfinder-z-range=-3000.:3000.
           --seedfinder-r-range=33.:320.
           --seedfinder-vertex-range=-200.:200.
           --seedfinder-minPt=0.9
           --seedfinder-cotThetaMax=27.2899
           --seedfinder-deltaR-range=20.:200.
           --seedfinder-impactMax=10.
           --seedfinder-sigmaScattering=3.
           --seedfinder-maxPtScattering=10.
           --seedfinder-maxSeedsPerSpM=1
           --max-num-branches-per-seed=3
           --max-num-branches-per-surface=5
           --track-candidates-range=7:20
           --min-step-length-for-next-surface=0.5
           --max-step-counts-for-next-surface=100
           --chi2-max=10.
           --max-num-skipping-per-cand=2
           --stepping-min-stepsize=0.0001
           --rk-tolerance-mm=0.0001
           --stepping-path-limit=5.
           --stepping-max-rk-updates=10000
           --stepping-use-mean-loss=1
           --stepping-use-eloss-gradient=0
           --stepping-use-field-gradient=0
           --stepping-do-covariance-transport=1
           --overstep-tolerance-um=-300.
           --min-mask-tolerance-mm=0.00001
           --max-mask-tolerance-mm=3.
           --search-window=0:0)

G100_CUTS=${G200_CUTS[@]}
G100_CUTS+=(--reco-stage=seeding)

# Select which of these flags to use.
TRACCC_CUTS=()
if [[ "${TRACCC_THROUGPUT_TYPE}" == "g200" ]]; then
   TRACCC_CUTS=${G200_CUTS[@]}
elif [[ "${TRACCC_THROUGPUT_TYPE}" == "g100" ]]; then
   TRACCC_CUTS=${G100_CUTS[@]}
elif [[ "${TRACCC_THROUGPUT_TYPE}" != "traccc" ]]; then
   echo "***"
   echo "*** Unknown throughput type: '${TRACCC_THROUGPUT_TYPE}'"
   echo "***"
   exit 1
fi

# The input directories to use.
TRACCC_INPUT_DIRS=("ttbar_mu140/hits" "ttbar_mu200/hits")

# Put a header on the CSV file.
echo "directory,threads,loaded_events,cold_run_events,processed_events,warm_up_time,processing_time" \
   > "${TRACCC_CSV_FILE}"

# Counter for a nice printout.
COUNTER=1
COUNT=$((${#TRACCC_INPUT_DIRS[@]}*${TRACCC_MAX_THREADS}*${TRACCC_REPETITIONS}))

# Iterate over the number of threads.
for NTHREAD in $(seq ${TRACCC_MIN_THREADS} ${TRACCC_THREAD_STEP} ${TRACCC_MAX_THREADS}); do
   # Iterate over the input datasets.
   for EVTDIR in ${TRACCC_INPUT_DIRS[@]}; do
      # Perform the requested number of repetitions.
      for REPEAT in $(seq ${TRACCC_REPETITIONS}); do

         # Tell the user what's happening.
         echo ""
         echo "Running test ${COUNTER} / ${COUNT}"
         ((COUNTER++))

         # Run the throughput test.
         ${TRACCC_EXECUTABLE}                                                  \
            --detector-file="${TRACCC_INPUT_DIR}/ITk_DetectorBuilder_geometry.json" \
            --material-file="${TRACCC_INPUT_DIR}/ITk_detector_material.json"   \
            --grid-file="${TRACCC_INPUT_DIR}/ITk_DetectorBuilder_surface_grids.json" \
            --digitization-file="${TRACCC_INPUT_DIR}/ITk_digitization_config_with_strips_with_shift_annulus_flip.json" \
            --read-bfield-from-file                                            \
            --bfield-file="${TRACCC_INPUT_DIR}/ITk_bfield.cvf"                 \
            --input-directory="${TRACCC_INPUT_DIR}/${EVTDIR}/"                 \
            --use-acts-geom-source=0                                           \
            --input-events=100                                                 \
            --cpu-threads=${NTHREAD}                                           \
            --cold-run-events=$((5*${NTHREAD}))                                \
            --processed-events=$((100*${NTHREAD}))                             \
            --log-file="${TRACCC_CSV_FILE}"                                    \
            ${TRACCC_CUTS[@]}
      done
   done
done
