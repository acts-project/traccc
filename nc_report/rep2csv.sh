#!/usr/bin/env bash

ncu \
  --import ./nc_report.ncu-rep \
  --csv \
  --page raw \
  --log-file ./nc_report.csv
