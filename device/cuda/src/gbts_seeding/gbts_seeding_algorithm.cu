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

	//node making
	unsigned int* d_layerCounts{};
	unsigned short* d_spacepointsLayer{};
	//begin_idx + 1 for the surfaceToLayerMap or -layerBin if one to one
	short* d_volumeToLayerMap{};	
	//surface_id, layerBin
	uint2* d_surfaceToLayerMap{};
	char* d_layerIsEndcap{};
	
	//x,y,z,cluster width in eta
	float4* d_reducedSP{};
	//output of layer binning
	float4* d_sp_params{};	
	
	int4* d_layerInfo{};
	float2* d_layerGeo{};

	//GraphMaking

	//GraphProccessing

	//output
};

gbts_seeding_algorithm::gbts_seeding_algorithm(const gbts_seedfinder_config& cfg, traccc::memory_resource& mr, vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
                                               : messaging(logger->clone()), m_config(cfg), m_mr(mr), m_copy(copy), m_stream(str) {}

gbts_seeding_algorithm::output_type gbts_seeding_algorithm::operator()(const traccc::edm::spacepoint_collection::const_view& spacepoints, const traccc::measurement_collection_types::const_view& measurements) const {
	
    edm::seed_collection::buffer output_seeds(0, m_mr.main, vecmem::data::buffer_type::resizable);

	gbts_ctx ctx;

	cudaStream_t stream = details::get_stream(m_stream);
	
	//0. bin spacepoints by layer(disk) or any other maping supplied to the config.m_surfaceToLayerMap
	ctx.nSp = m_copy.get().get_size(spacepoints); //why is get needed?
	if(ctx.nSp == 0) return output_seeds;

	unsigned int nThreads = 1024;
	unsigned int nBlocks = 1+(ctx.nSp-1)/nThreads;
	
	cudaMalloc(&ctx.d_layerCounts, m_config.nLayers*sizeof(unsigned int));
	cudaMemset(ctx.d_layerCounts, 0 ,m_config.nLayers*sizeof(unsigned int));	
	
	cudaMalloc(&ctx.d_spacepointsLayer, ctx.nSp*sizeof(unsigned char));	
	cudaMalloc(&ctx.d_reducedSP, ctx.nSp*sizeof(float4));	
	
	cudaMalloc(&ctx.d_volumeToLayerMap, sizeof(short)*m_config.volumeMapSize);		
	cudaMemcpyAsync(ctx.d_volumeToLayerMap, m_config.volumeToLayerMap.get(), sizeof(short)*m_config.volumeMapSize, cudaMemcpyHostToDevice, stream);
	
	if(m_config.surfaceMapSize != 0) {
		cudaMalloc(&ctx.d_surfaceToLayerMap, sizeof(uint2)*m_config.surfaceMapSize);	
		cudaMemcpyAsync(ctx.d_surfaceToLayerMap, m_config.surfaceToLayerMap.data(), sizeof(uint2)*m_config.surfaceMapSize, cudaMemcpyHostToDevice, stream);
	} //may be zero and correct, volumeMapSize and nLayers are checked at config

	cudaMalloc(&ctx.d_layerIsEndcap, sizeof(char)*m_config.nLayers);
	cudaMemcpyAsync(ctx.d_layerIsEndcap, m_config.layerInfo.isEndcap.data(), sizeof(char)*m_config.nLayers, cudaMemcpyHostToDevice, stream);	
	
	kernels::count_sp_by_layer<<<nBlocks,nThreads,0,stream>>>(spacepoints,measurements,
								ctx.d_volumeToLayerMap,ctx.d_surfaceToLayerMap,ctx.d_layerIsEndcap, 
                                ctx.d_reducedSP, ctx.d_layerCounts, ctx.d_spacepointsLayer,
								ctx.nSp, m_config.volumeMapSize, m_config.surfaceMapSize);

	cudaFree(ctx.d_volumeToLayerMap);
	cudaFree(ctx.d_surfaceToLayerMap);
	cudaFree(ctx.d_layerIsEndcap);

	//prefix sum layerCounts
	std::unique_ptr<unsigned int[]> layerCounts = std::make_unique<unsigned int[]>(m_config.nLayers+1);

	cudaMemcpyAsync(layerCounts.get(), ctx.d_layerCounts, m_config.nLayers*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);	
	for(int layer = 0; layer<m_config.nLayers; layer++) TRACCC_INFO("layer " << layer << " with " << layerCounts[layer] << "sp");
	for(int layer = 1; layer<m_config.nLayers; layer++) {
		layerCounts[layer] += layerCounts[layer-1];
	}
	TRACCC_INFO("sum layer sp " << layerCounts[m_config.nLayers-1]);
	cudaMemcpyAsync(ctx.d_layerCounts, layerCounts.get(), m_config.nLayers*sizeof(unsigned int), cudaMemcpyHostToDevice, stream);	

	cudaMalloc(&ctx.d_sp_params, ctx.nSp*sizeof(float4));	

	kernels::bin_sp_by_layer<<<nBlocks, nThreads, 0, stream>>>(ctx.d_sp_params, ctx.d_reducedSP, ctx.d_layerCounts, ctx.d_spacepointsLayer, ctx.nSp);	

	cudaFree(ctx.d_reducedSP);
	cudaFree(ctx.d_spacepointsLayer);

	//1. histogram spacepoints by layer->eta->phi and convert to nodes phi,r,z,tau_min,tau_max
	//do this in config setup?
	cudaMalloc(&ctx.d_layerInfo, sizeof(int2)*m_config.nLayers);
	cudaMemcpyAsync(ctx.d_layerInfo, m_config.layerInfo.info.data(), sizeof(int2)*m_config.nLayers, cudaMemcpyHostToDevice, stream);	
	
	cudaMalloc(&ctx.d_layerGeo, sizeof(float2)*m_config.nLayers);
	cudaMemcpyAsync(ctx.d_layerGeo, m_config.layerInfo.geo.data(), sizeof(float2)*m_config.nLayers, cudaMemcpyHostToDevice, stream);	

	cudaFree(ctx.d_layerCounts);
	cudaFree(ctx.d_layerInfo);
	cudaFree(ctx.d_layerGeo);

	//2. Find edges between spacepoint pairs

	//3. Link edges into graph

	//4. Prune unlinked edges from graph

	//5. Find longest segments with CCA

	//6. extract seeds, longest segment first
	cudaFree(ctx.d_sp_params);

	return output_seeds;
}

} //namespace traccc::cuda
