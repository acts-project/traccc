#!/usr/bin/env bash
set -euo pipefail

# List of register limits to test
#REGS=(64 96 128 144 160 176 192 208 224 240 255)
REGS=(32 48 64 80 96 112 128 144 160 176 192 208 224 240 255)

# Paths
CMAKE_FILE="cmake/traccc-compiler-options-cuda.cmake"
BUILD_SCRIPT="./01_build.sh"
RUN_SCRIPT="./02_run.sh"
LOG_FILE="log/maxreg_iter.log"

# Backup original CMake file and clear previous log
cp "${CMAKE_FILE}" "${CMAKE_FILE}.bak"
: > "${LOG_FILE}"

for r in "${REGS[@]}"; do
  echo "===== --maxrregcount=${r} =====" | tee -a "${LOG_FILE}"

  # Replace the number in the CMake flag
  sed -i -E "s#(--maxrregcount=)[0-9]+#\1${r}#g" "${CMAKE_FILE}"

  # Build; on failure, note it and move on
  if ! "${BUILD_SCRIPT}"; then
    echo "BUILD FAILED for ${r}" | tee -a "${LOG_FILE}"
    continue
  fi

  # Run and capture the last line
  LAST_LINE=$("${RUN_SCRIPT}" | tail -n1)
  echo "${LAST_LINE}" | tee -a "${LOG_FILE}"
done

# Restore original CMake file
mv "${CMAKE_FILE}.bak" "${CMAKE_FILE}"

echo "Done. Results written to ${LOG_FILE}"

