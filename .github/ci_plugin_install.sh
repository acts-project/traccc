#!/bin/bash
#
# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2024 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0
#
# This is a (hopefully) temporary script for installing the CodePlay oneAPI
# plugins for the SYCL tests.
#

# The platform name.
PLATFORM_NAME=$1

# Do the installation(s) when on a SYCL platform, and with the installation
# being under /opt/intel/oneapi.
if [ "${PLATFORM_NAME}" = "SYCL" ]; then
   if [ -d "/opt/intel/oneapi" ]; then
      curl -SL "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=nvidia&version=2024.1.0" \
           -o nvidia_plugin.sh
      sh nvidia_plugin.sh -i /opt/intel/oneapi -y
      rm nvidia_plugin.sh
      curl -SL "https://developer.codeplay.com/api/v1/products/download?product=oneapi&variant=amd&version=2024.1.0" \
           -o amd_plugin.sh
      sh amd_plugin.sh -i /opt/intel/oneapi -y
      rm amd_plugin.sh
   fi
fi
