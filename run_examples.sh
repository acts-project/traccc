#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

build_dir=$1

export TRACCC_TEST_DATA_DIR="$SCRIPT_DIR/data"


set -e
set -x

$build_dir/bin/ccl_example $TRACCC_TEST_DATA_DIR/tml_pixels/event000000000-cells.csv
$build_dir/bin/io_dec_par_example tml_detector/trackml-detector.csv tml_pixels 10
$build_dir/bin/seq_example tml_detector/trackml-detector.csv tml_pixels 10
$build_dir/bin/par_example tml_detector/trackml-detector.csv tml_pixels 10
$build_dir/bin/seq_single_module