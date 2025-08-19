/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "../utils/cuda_error_handling.hpp"
#include "../utils/utils.hpp"
#include "./kernels/GbtsNodesMakingKernels.cuh"
//#include "./kernels/GbtsGraphMaking.cuh"
//#include "./kernels/GbtsGraphProcessing.cuh"

#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"

namespace traccc::cuda {

struct gbts_ctx {
	//counters
	unsigned int nSp{};
	unsigned int nEdges{};
	unsigned int nConnectedEdges{};
	unsigned int nSeeds{};
	//nEdges, nConnections, nConnectedEdges, .., nSeeds
	unsigned int* d_counters;	

	//NodeMaking
	unsigned int* d_layerCounts{};
	unsigned short* d_spacepointsLayer{};
	//begin_idx + 1 for the surfaceToLayerMap or -layerBin if one to one
	int* d_volumeToLayerMap{};	
	//surface_id, layerBin
	uint2* d_surfaceToLayerMap{};

	traccc::device::gbts_layerInfo* d_layerInfo{};

	//x,y,z,cluster width in eta
	float4* d_reducedSP{};
	//output of layer binning
	float4* d_sp_params{};	

	//GraphMaking

	//GraphProccessing

	//output
};

gbts_seeding_algorithm::gbts_seeding_algorithm(const gbts_seedfinder_config& cfg, traccc::memory_resource& mr, vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
                                               : messaging(logger->clone()), m_config(cfg), m_mr(mr), m_copy(copy), m_stream(str) {}

gbts_seeding_algorithm::output_type gbts_seeding_algorithm::operator()(const traccc::edm::spacepoint_collection::const_view& spacepoints, const traccc::measurement_collection_types::const_view& measurements) const {

	gbts_seeding_algorithm::output_type output_seeds;

	gbts_ctx ctx;

	cudaStream_t stream = details::get_stream(m_stream);

	//0. bin spacepoints by layer(disk) or any other maping supplied to the config.m_surfaceToLayerMap
	ctx.nSp = m_copy.get().get_size(spacepoints); //why is get needed?

	unsigned int nThreads = 1024;
	unsigned int nBlocks = 1+(ctx.nSp-1)/nThreads;

	cudaMalloc(&ctx.d_layerCounts, m_config.nLayers*sizeof(unsigned int));	
	cudaMalloc(&ctx.d_spacepointsLayer, ctx.nSp*sizeof(unsigned char));	
	cudaMalloc(&ctx.d_reducedSP, ctx.nSp*sizeof(float4));	

	cudaMalloc(&ctx.d_volumeToLayerMap, sizeof(int)*m_config.maxVolIndex);	
	cudaMalloc(&ctx.d_surfaceToLayerMap, sizeof(uint2)*m_config.surfaceMapSize);	

	cudaMemcpyAsync(ctx.d_volumeToLayerMap, m_config.volumeToLayerMap.get(), sizeof(int)*m_config.maxVolIndex, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(ctx.d_surfaceToLayerMap, m_config.surfaceToLayerMap.data(), sizeof(uint2)*m_config.surfaceMapSize, cudaMemcpyHostToDevice, stream);

	cudaMalloc(&ctx.d_layerInfo, sizeof(traccc::device::gbts_layerInfo)*m_config.nLayers);
	cudaMemcpyAsync(ctx.d_layerInfo, m_config.layerInfo.data(), sizeof(traccc::device::gbts_layerInfo)*m_config.nLayers, cudaMemcpyHostToDevice, stream);	

	kernels::count_sp_by_layer<<<nBlocks,nThreads,0,stream>>>(spacepoints,measurements,
	                            ctx.d_volumeToLayerMap,ctx.d_surfaceToLayerMap,ctx.d_layerInfo, 
                                ctx.d_reducedSP, ctx.d_layerCounts, ctx.d_spacepointsLayer,
								ctx.nSp, m_config.surfaceMapSize);
	//prefix sum layerCounts
	std::unique_ptr<unsigned int[]> layerCounts = std::make_unique<unsigned int[]>(m_config.nLayers);

	cudaMemcpyAsync(ctx.d_layerCounts, layerCounts.get(), m_config.nLayers*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);	
	for(int layer = 1; layer<m_config.nLayers; layer++) {
		layerCounts[layer] += layerCounts[layer-1];
	}
	cudaMemcpyAsync(layerCounts.get(), ctx.d_layerCounts, m_config.nLayers*sizeof(unsigned int), cudaMemcpyHostToDevice, stream);	

	cudaMalloc(&ctx.d_sp_params, ctx.nSp*sizeof(float4));	

	kernels::bin_sp_by_layer<<<nBlocks, nThreads, 0, stream>>>(ctx.d_sp_params, ctx.d_reducedSP, ctx.d_layerCounts, ctx.d_spacepointsLayer, ctx.nSp);	

	//1. histogram spacepoints by layer->eta->phi and convert to nodes phi,r,z,tau_min,tau_max

	//2. Find edges between spacepoint pairs

	//3. Link edges into graph

	//4. Prune unlinked edges from graph

	//5. Find longest segments with CCA

	//6. extract seeds, longest segment first

	return output_seeds;
}

} //namespace traccc::cuda
