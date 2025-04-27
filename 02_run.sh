#!/usr/bin/env bash

./build/bin/traccc_throughput_mt_cuda \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --use-detray-detector=on \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=on \
  --input-directory=odd/geant4_10muon_10GeV/ \
  --input-events=10 \
  --processed-events=1000 \
  --threads=1 \
  
