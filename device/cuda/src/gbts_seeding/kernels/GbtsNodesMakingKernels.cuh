/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

//cuda includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_functions.h>

//Project includes
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/measurement.hpp"

// Detray include(s).
#include <detray/geometry/barcode.hpp>

namespace traccc::cuda::kernels {
//just pixel spacepoints for now

__global__ void count_sp_by_layer(const traccc::edm::spacepoint_collection::const_view spacepoints_view, const traccc::measurement_collection_types::const_view measurements_view, 
                                  const short* volumeToLayerMap, const uint2* surfaceToLayerMap, const char* d_layerIsEndcap, 
                                  float4* reducedSP, unsigned int* layerCounts, short* spacepointsLayer,
                                  const unsigned int nSp, const unsigned int volumeMapSize, const unsigned int surfaceMapSize) {
	//shared mem volumeToLayer map
	const traccc::measurement_collection_types::const_device measurements(measurements_view);
	const traccc::edm::spacepoint_collection::const_device spacepoints(spacepoints_view);
	
	for(int spIdx = threadIdx.x + blockDim.x*blockIdx.x; spIdx<spacepoints.size(); spIdx += blockDim.x*gridDim.x) {
		//get the layer of the spacepoint
		const traccc::edm::spacepoint_collection::const_device::const_proxy_type spacepoint = spacepoints.at(spIdx);	
		const traccc::measurement measurement = measurements.at(spacepoint.measurement_index_1());
		
		detray::geometry::barcode barcode = measurement.surface_link;	
			
		//some volume_ids map one to one with layer others need searching
		if(barcode.volume() > volumeMapSize) continue; //unconfigured volume
		
		short begin_or_bin = volumeToLayerMap[barcode.volume()];
		if(begin_or_bin == SHRT_MAX) continue; //unconfigured volume 		

		unsigned int layerIdx;
		if(begin_or_bin < 0) {
			unsigned int surface_index = static_cast<unsigned int>(barcode.index());
			for(unsigned int surface = -1*(begin_or_bin + 1); surface < surfaceMapSize;surface++) {
				uint2 surfaceBinPair = surfaceToLayerMap[surface];
				if(surfaceBinPair.x == surface_index) { 
					layerIdx = surfaceBinPair.y; 
					break;
				}
			}
		}
		else layerIdx = static_cast<unsigned int>(begin_or_bin);
		
		float cluster_diameter = measurement.diameter;
		if(d_layerIsEndcap[layerIdx] == 1) { //get info from sp -> move to before layerIdx map?
			if(cluster_diameter > 0.2) {
				spacepointsLayer[spIdx] = -1;
				continue;
			}
			cluster_diameter = -1; //flag for skiping tau range calculation
		}
		//count and store x,y,z,cw info
		atomicAdd(&layerCounts[layerIdx], 1);
		spacepointsLayer[spIdx] = layerIdx;
		const traccc::point3 pos = spacepoint.global();
		reducedSP[spIdx] = make_float4(pos[0], pos[1], pos[2], cluster_diameter);
	}
}

//layerCounts is prefix sumed on CPU inbetween count_sp_by_layer and this kerenel
__global__ void bin_sp_by_layer(float4* sp_params ,float4* reducedSP, unsigned int* layerCounts, short* spacepointsLayer, int* original_sp_idx, const unsigned int nSp) {
	for(int spIdx = threadIdx.x + blockDim.x*blockIdx.x; spIdx<nSp; spIdx += blockDim.x*gridDim.x) {
		short layerIdx = spacepointsLayer[spIdx];
		if(layerIdx == -1) continue; 
		unsigned int binedIdx = atomicSub(&layerCounts[layerIdx], 1) - 1;
		original_sp_idx[binedIdx] = spIdx;
		sp_params[binedIdx] = reducedSP[spIdx];
	}
}

}
