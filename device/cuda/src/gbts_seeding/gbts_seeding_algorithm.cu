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
#include "./kernels/GbtsGraphMakingKernels.cuh"
//#include "./kernels/GbtsGraphProcessingKernels.cuh"

#include "traccc/cuda/gbts_seeding/gbts_seeding_algorithm.hpp"

namespace traccc::cuda {

struct gbts_ctx {
	//counters
	unsigned int nSp{};
	unsigned int nNodes{};
	unsigned int nUsedBinPairs{};
	unsigned int nMaxEdges{};

	unsigned int nEdges{};
	unsigned int nConnections{};
	unsigned int nConnectedEdges{};
	unsigned int nSeeds{};
	//nEdges, nConnections, nConnectedEdges, .., nSeeds
	unsigned int* d_counters;	
	
	//device cut values
	gbts_algo_params* d_algo_params;

	//node making and binning
	unsigned int* d_layerCounts{};
	short* d_spacepointsLayer{}; 
	short* d_volumeToLayerMap{}; //begin_idx + 1 for the surfaceToLayerMap or -layerBin if one to one
	uint2* d_surfaceToLayerMap{}; //surface_index, layerBin
	char* d_layerIsEndcap{};
	int* d_original_sp_idx{}; // conversion to original sp from post layer binning index
	int* d_node_index{}; // conversion to orignal sp/node index from post binning index
	
	//x,y,z,cluster width in eta
	float4* d_reducedSP{};
	//layer binned reducedSP
	float4* d_sp_params{};	
	
	int2* d_layer_info{};
	float2* d_layer_geo{};
	
	int* d_node_eta_index{};
	int* d_node_phi_index{};

	unsigned int* d_eta_phi_histo{};//for data binning
	unsigned int* d_phi_cusums{};//for data binning
	unsigned int* d_eta_node_counter{};//for data binning

	int2* d_eta_bin_views{};//views of the nodes
	std::unique_ptr<int[]>   h_eta_bin_views{};//eta-bin views of the node_params array
	
	float2* d_bin_rads{};//minimum and maximum r of nodes inside an eta-bin
	std::unique_ptr<float[]> h_bin_rads{};

	uint4* d_bin_pair_views{};
	std::unique_ptr<unsigned int[]> h_bin_pair_views{};

	std::unique_ptr<float[]> h_bin_pair_dphi{};
	float* d_bin_pair_dphi{};
	//node making output
	float* d_node_params{};

	//GraphMaking
	int2* d_edge_nodes{};
	kernels::half4* d_edge_params{};	

	unsigned int* d_num_incoming_edges{}; 
	int* d_edge_links{};

	unsigned char* d_num_neighbours{};
	int* d_reIndexer{};
	int* d_neighbours{};
	int* d_output_graph{};

	//GraphProccessing

	//output
};

gbts_seeding_algorithm::gbts_seeding_algorithm(const gbts_seedfinder_config& cfg, traccc::memory_resource& mr, vecmem::copy& copy, stream& str, std::unique_ptr<const Logger> logger)
                                               : messaging(logger->clone()), m_config(cfg), m_mr(mr), m_copy(copy), m_stream(str) {}

gbts_seeding_algorithm::output_type gbts_seeding_algorithm::operator()(const traccc::edm::spacepoint_collection::const_view& spacepoints, const traccc::measurement_collection_types::const_view& measurements) const {
	
    edm::seed_collection::buffer output_seeds(0, m_mr.main, vecmem::data::buffer_type::resizable);

	gbts_ctx ctx;

	cudaStream_t stream = details::get_stream(m_stream);
	
	cudaMalloc(&ctx.d_algo_params, sizeof(m_config.algo_params));
	cudaMemcpyAsync(ctx.d_algo_params, &m_config.algo_params , sizeof(m_config.algo_params), cudaMemcpyHostToDevice);


	//0. bin spacepoints by layer(disk) or any other maping supplied to the config.m_surfaceToLayerMap
	ctx.nSp = m_copy.get().get_size(spacepoints); //why is get needed?
	if(ctx.nSp == 0) return output_seeds;

	unsigned int nThreads = 1024;
	unsigned int nBlocks = 1+(ctx.nSp-1)/nThreads;
	
	cudaMalloc(&ctx.d_layerCounts, (m_config.nLayers+1)*sizeof(unsigned int));
	cudaMemset(ctx.d_layerCounts, 0 , (m_config.nLayers+1)*sizeof(unsigned int));	
	
	cudaMalloc(&ctx.d_spacepointsLayer, ctx.nSp*sizeof(short));	
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

	cudaStreamSynchronize(stream);
	
	cudaFree(ctx.d_volumeToLayerMap);
	cudaFree(ctx.d_surfaceToLayerMap);
	cudaFree(ctx.d_layerIsEndcap);

	//prefix sum layerCounts
	std::unique_ptr<unsigned int[]> layerCounts = std::make_unique<unsigned int[]>(m_config.nLayers+1);

	cudaMemcpyAsync(layerCounts.get(), ctx.d_layerCounts, (m_config.nLayers+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);	
	for(int layer = 1; layer < m_config.nLayers + 1; layer++) {
		layerCounts[layer] += layerCounts[layer-1];
	}
	cudaMemcpyAsync(ctx.d_layerCounts, layerCounts.get(), m_config.nLayers*sizeof(unsigned int), cudaMemcpyHostToDevice, stream);	
	ctx.nNodes = layerCounts[m_config.nLayers];
	if(ctx.nNodes == 0) return output_seeds;
	layerCounts.reset();
	
	cudaMalloc(&ctx.d_sp_params, ctx.nSp*sizeof(float4));	
	cudaMalloc(&ctx.d_original_sp_idx, ctx.nSp*sizeof(int));	

	kernels::bin_sp_by_layer<<<nBlocks, nThreads, 0, stream>>>(ctx.d_sp_params, ctx.d_reducedSP, ctx.d_layerCounts, ctx.d_spacepointsLayer, ctx.d_original_sp_idx, ctx.nSp);	
   
	cudaStreamSynchronize(stream);
	cudaError_t error = cudaGetLastError();

	if(error != cudaSuccess) {
		TRACCC_ERROR("spacepoint layer binning: CUDA error: " << cudaGetErrorString(error));
		return output_seeds;
	}

	cudaFree(ctx.d_spacepointsLayer);

	//1. histogram spacepoints by layer->eta->phi and convert to nodes phi,r,z,tau_min,tau_max
	//do this in config setup?
	cudaMalloc(&ctx.d_layer_info, sizeof(int2)*m_config.nLayers);
	cudaMemcpyAsync(ctx.d_layer_info, m_config.layerInfo.info.data(), sizeof(int2)*m_config.nLayers, cudaMemcpyHostToDevice, stream);	
	
	cudaMalloc(&ctx.d_layer_geo, sizeof(float2)*m_config.nLayers);
	cudaMemcpyAsync(ctx.d_layer_geo, m_config.layerInfo.geo.data(), sizeof(float2)*m_config.nLayers, cudaMemcpyHostToDevice, stream);	
	
	cudaMalloc(&ctx.d_node_phi_index, sizeof(int)*ctx.nNodes);

	nThreads = 256;
	int nNodesPerBlock = nThreads*64;
     
	nBlocks = 1+(ctx.nNodes-1)/nNodesPerBlock;
   
	kernels::node_phi_binning_kernel<<<nBlocks, nThreads, 0, stream>>>(ctx.d_sp_params, ctx.d_node_phi_index, nNodesPerBlock, ctx.nNodes, m_config.n_phi_bins);

	cudaStreamSynchronize(stream);
	
	cudaMalloc(&ctx.d_node_eta_index, sizeof(int)*ctx.nNodes);

	nBlocks = m_config.nLayers;

	kernels::node_eta_binning_kernel<<<nBlocks, nThreads, 0, stream>>>(ctx.d_sp_params, ctx.d_layer_info, ctx.d_layer_geo, ctx.d_node_eta_index, ctx.d_layerCounts ,m_config.nLayers);
	
	cudaStreamSynchronize(stream);
	
	cudaFree(ctx.d_layerCounts);
	cudaFree(ctx.d_layer_info);
	cudaFree(ctx.d_layer_geo);
	
	error = cudaGetLastError();

	if(error != cudaSuccess) {
		TRACCC_ERROR("eta-phi binning: CUDA error: " << cudaGetErrorString(error));
		return output_seeds;
	}
	
	unsigned int hist_size = static_cast<unsigned int>(sizeof(unsigned int))*m_config.n_eta_bins*m_config.n_phi_bins;
	cudaMalloc(&ctx.d_eta_phi_histo, hist_size);
	cudaMemset(ctx.d_eta_phi_histo, 0, hist_size); 
	cudaMalloc(&ctx.d_phi_cusums, hist_size);	

    nBlocks = 1 + (ctx.nNodes-1)/nNodesPerBlock;

    kernels::eta_phi_histo_kernel<<<nBlocks, nThreads, 0, stream>>>(ctx.d_node_phi_index, ctx.d_node_eta_index, ctx.d_eta_phi_histo, nNodesPerBlock, ctx.nNodes, m_config.n_phi_bins);

    cudaStreamSynchronize(stream);
 
    error = cudaGetLastError();

    if(error != cudaSuccess) {
        TRACCC_ERROR("eta-phi histo: CUDA error: " << cudaGetErrorString(error));
        return output_seeds;
    }
	
	cudaMalloc(&ctx.d_eta_node_counter, sizeof(unsigned int)*m_config.n_eta_bins);

    int nBinsPerBlock = 128;
        
    nThreads = nBinsPerBlock;

    nBlocks = 1 + (m_config.n_eta_bins - 1)/nBinsPerBlock;

    kernels::eta_phi_counting_kernel<<<nBlocks, nThreads, 0, stream>>>(ctx.d_eta_phi_histo, ctx.d_eta_node_counter, ctx.d_phi_cusums, nBinsPerBlock, m_config.n_eta_bins, m_config.n_phi_bins);

    cudaStreamSynchronize(stream);
	cudaFree(ctx.d_eta_phi_histo);	

    error = cudaGetLastError();

    if(error != cudaSuccess) {
        TRACCC_ERROR("eta-phi counting: CUDA error: " << cudaGetErrorString(error));
        return output_seeds;
    }
	
	std::unique_ptr<unsigned int[]> eta_sums = std::make_unique<unsigned int[]>(m_config.n_eta_bins);

	cudaMemcpyAsync(&eta_sums[0], &ctx.d_eta_node_counter[0], sizeof(unsigned int)*m_config.n_eta_bins, cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);

	for(int k=1;k<m_config.n_eta_bins;k++) eta_sums[k] += eta_sums[k-1];

	//send back
	cudaMemcpyAsync(&ctx.d_eta_node_counter[0], &eta_sums[0], sizeof(unsigned int)*m_config.n_eta_bins, cudaMemcpyHostToDevice, stream);

	ctx.h_eta_bin_views = std::make_unique<int[]>(2*m_config.n_eta_bins);

	for(int view_idx = 0; view_idx < m_config.n_eta_bins; view_idx++) {
	   int pos = 2*view_idx;
	   ctx.h_eta_bin_views[pos]   = (view_idx == 0) ? 0 : eta_sums[view_idx-1];
	   ctx.h_eta_bin_views[pos+1] = eta_sums[view_idx];
	} 
	eta_sums.reset();

	cudaStreamSynchronize(stream);

	kernels::eta_phi_prefix_sum_kernel<<<nBlocks, nThreads, 0, stream>>>(ctx.d_eta_node_counter, ctx.d_phi_cusums, nBinsPerBlock, m_config.n_eta_bins, m_config.n_phi_bins);

	cudaStreamSynchronize(stream);
    cudaFree(ctx.d_eta_node_counter);

	error = cudaGetLastError();

	if(error != cudaSuccess) {
	   TRACCC_ERROR("eta-phi cusum: CUDA error: " << cudaGetErrorString(error));
	   return output_seeds;
	}
		
	cudaMalloc(&ctx.d_node_params, 5*sizeof(float)*ctx.nNodes);
	cudaMalloc(&ctx.d_node_index, sizeof(int)*ctx.nNodes);

	nThreads = 256;
	nNodesPerBlock = nThreads*64;
	   
	nBlocks = 1 + (ctx.nNodes-1)/nNodesPerBlock;

	kernels::node_sorting_kernel<<<nBlocks, nThreads, 0, stream>>>(ctx.d_sp_params, ctx.d_node_eta_index, ctx.d_node_phi_index, 
														           ctx.d_phi_cusums, ctx.d_node_params, ctx.d_node_index, ctx.d_original_sp_idx,
                                                                   nNodesPerBlock, ctx.nNodes, m_config.n_phi_bins);

	cudaStreamSynchronize(stream);
	cudaFree(ctx.d_sp_params);
	cudaFree(ctx.d_original_sp_idx);
    cudaFree(ctx.d_phi_cusums);
	cudaFree(ctx.d_node_eta_index);
	cudaFree(ctx.d_node_phi_index);

	error = cudaGetLastError();

	if(error != cudaSuccess) {
	   TRACCC_ERROR("node sorting: CUDA error: " << cudaGetErrorString(error));
	   return output_seeds;
	}
	
	cudaMalloc(&ctx.d_eta_bin_views, sizeof(int2)*m_config.n_eta_bins);
	cudaMalloc(&ctx.d_bin_rads, sizeof(float2)*m_config.n_eta_bins);

	cudaMemcpyAsync(&ctx.d_eta_bin_views[0], ctx.h_eta_bin_views.get(), 2*m_config.n_eta_bins*sizeof(int), cudaMemcpyHostToDevice, stream);

	cudaStreamSynchronize(stream);

	nBinsPerBlock = 128;
	   
	nThreads = nBinsPerBlock;

	nBlocks = 1 + (m_config.n_eta_bins-1)/nBinsPerBlock;

	kernels::minmax_rad_kernel<<<nBlocks, nThreads, 0, stream>>>(ctx.d_eta_bin_views, ctx.d_node_params,
															 ctx.d_bin_rads, nBinsPerBlock, m_config.n_eta_bins);

	cudaStreamSynchronize(stream);
	cudaFree(ctx.d_eta_bin_views);

	error = cudaGetLastError();

	if(error != cudaSuccess) {
	   TRACCC_ERROR("node sorting: CUDA error: " << cudaGetErrorString(error));
	   return output_seeds;
	}

	ctx.h_bin_rads = std::make_unique<float[]>(2*m_config.n_eta_bins);

	cudaMemcpyAsync(ctx.h_bin_rads.get(), &ctx.d_bin_rads[0], 2*sizeof(float)*m_config.n_eta_bins, cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);

	cudaFree(ctx.d_bin_rads);

	//2. prepare input for the graph making part of the code:

	unsigned int nBinPairs = 0;//the number of eta bin pairs

	for(std::pair<int, int> binPair : m_config.binTables) {//loop over bin pairs defined by the layer connection table and geometry settings

	   int bin1_begin = ctx.h_eta_bin_views[2*binPair.first];
	   int bin1_end   = ctx.h_eta_bin_views[2*binPair.first+1];

	   //large bins will be split into smaller sub-views
	   
	   unsigned int nNodesInBin1 = bin1_end - bin1_begin;

	   nBinPairs += 1 + (nNodesInBin1-1)/traccc::device::node_buffer_length;
	}

	ctx.h_bin_pair_views = std::make_unique<unsigned int[]>(4*nBinPairs);
	ctx.h_bin_pair_dphi  = std::make_unique<float[]>(nBinPairs);

	int pairIdx = 0;
	for(std::pair<int, int> binPair : m_config.binTables) {
	   
	   float rb1 = ctx.h_bin_rads[2*binPair.first];//min radius

	   unsigned int begin_bin1 = ctx.h_eta_bin_views[2*binPair.first];
	   unsigned int end_bin1    = ctx.h_eta_bin_views[2*binPair.first+1];
	   //skip empty pairs
	   if(begin_bin1 == end_bin1) continue;
	   if(ctx.h_eta_bin_views[2*binPair.second] == ctx.h_eta_bin_views[2*binPair.second+1]) continue;

	   float rb2 = ctx.h_bin_rads[2*binPair.second+1];//max radius
	   
	   float maxDeltaR = std::fabs(rb2 - rb1);// max radius of bin2 - min radius of bin1
			   
	   float deltaPhi = m_config.algo_params.min_delta_phi + m_config.algo_params.dphi_coeff*maxDeltaR;
	   if(maxDeltaR < 60) deltaPhi = m_config.algo_params.min_delta_phi_low_dr + m_config.algo_params.dphi_coeff_low_dr*maxDeltaR;

	   //splitting large bins into more consistent sizes
			   
	   unsigned int currBegin_bin1 = begin_bin1;

	   unsigned int currEnd_bin1 = end_bin1 < traccc::device::node_buffer_length ? end_bin1 : begin_bin1 + traccc::device::node_buffer_length;
	   
	   for(;currEnd_bin1 < end_bin1; currEnd_bin1 += traccc::device::node_buffer_length, pairIdx++) {
		   unsigned int offset = 4*pairIdx;
		   
		   ctx.h_bin_pair_views[offset] = currBegin_bin1;
		   ctx.h_bin_pair_views[1 + offset] = currEnd_bin1;
		   ctx.h_bin_pair_views[2 + offset] = ctx.h_eta_bin_views[2*binPair.second];
		   ctx.h_bin_pair_views[3 + offset] = ctx.h_eta_bin_views[2*binPair.second + 1];
		   ctx.h_bin_pair_dphi[pairIdx]     = deltaPhi;
						   
		   currBegin_bin1 = currEnd_bin1;
	   }
	   currEnd_bin1 = end_bin1;
	   
	   unsigned int offset = 4*pairIdx;

	   ctx.h_bin_pair_views[offset]     = currBegin_bin1;
	   ctx.h_bin_pair_views[1 + offset] = currEnd_bin1;
	   ctx.h_bin_pair_views[2 + offset] = ctx.h_eta_bin_views[2*binPair.second];
	   ctx.h_bin_pair_views[3 + offset] = ctx.h_eta_bin_views[2*binPair.second + 1];
	   ctx.h_bin_pair_dphi[pairIdx]     = deltaPhi;
	   pairIdx++;
	   
	}
	ctx.nUsedBinPairs = pairIdx;
	if(pairIdx == 0) return output_seeds;
	ctx.h_eta_bin_views.reset();	

	// allocate memory and copy bin pair views and phi cuts to GPU

	size_t data_size = ctx.nUsedBinPairs*4*sizeof(unsigned int);
	   
	cudaMalloc(&ctx.d_bin_pair_views, data_size);
	cudaMemcpyAsync(&ctx.d_bin_pair_views[0], &ctx.h_bin_pair_views[0], data_size, cudaMemcpyHostToDevice, stream);

	data_size = ctx.nUsedBinPairs*sizeof(float);

	cudaMalloc(&ctx.d_bin_pair_dphi, data_size);
	cudaMemcpyAsync(&ctx.d_bin_pair_dphi[0], &ctx.h_bin_pair_dphi[0], data_size, cudaMemcpyHostToDevice, stream);

	cudaMalloc(&ctx.d_counters, sizeof(unsigned int)*12);
	cudaMemset(ctx.d_counters, 0, 12*sizeof(unsigned int));

	cudaStreamSynchronize(stream);
	
	//2. Find edges between spacepoint pairs
	ctx.nMaxEdges = 7*ctx.nNodes;
	cudaMalloc(&ctx.d_edge_params, sizeof(kernels::half4)*ctx.nMaxEdges);
	cudaMalloc(&ctx.d_edge_nodes, sizeof(int2)*ctx.nMaxEdges);
	cudaMalloc(&ctx.d_num_incoming_edges, sizeof(unsigned int)*(ctx.nNodes+1));


	nBlocks = ctx.nUsedBinPairs;
	nThreads = 128;

	kernels::graphEdgeMakingKernel_ITk<<<nBlocks, nThreads, 0, stream>>>(ctx.d_bin_pair_views,
													ctx.d_bin_pair_dphi, ctx.d_node_params,
													ctx.d_algo_params, ctx.d_counters, ctx.d_edge_nodes, 
													ctx.d_edge_params, ctx.d_num_incoming_edges, ctx.nMaxEdges, m_config.n_phi_bins);

	cudaStreamSynchronize(stream);
	cudaFree(ctx.d_node_params);
	cudaFree(ctx.d_bin_pair_views);
	cudaFree(ctx.d_bin_pair_dphi);	

	error = cudaGetLastError();

	if(error != cudaSuccess) {
	   TRACCC_ERROR("edge making: CUDA error: " << cudaGetErrorString(error));
	   return output_seeds;
	}

	cudaMemcpyAsync(&ctx.nEdges, ctx.d_counters, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);

	TRACCC_INFO("Created " << ctx.nEdges << " edges under a cap of " << ctx.nMaxEdges);
	   
	if(ctx.nEdges >= ctx.nMaxEdges) ctx.nEdges = ctx.nMaxEdges-1;
	else if(ctx.nEdges == 0) return output_seeds;

	std::unique_ptr<unsigned int[]> cusum = std::make_unique<unsigned int[]>(ctx.nNodes+1);

	data_size = (ctx.nNodes+1)*sizeof(unsigned int);

	cudaMemcpyAsync(&cusum[0], ctx.d_num_incoming_edges, data_size, cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);

	for(int k=0;k<ctx.nNodes;k++) cusum[k+1] += cusum[k];

	cudaMemcpyAsync(ctx.d_num_incoming_edges, &cusum[0], data_size, cudaMemcpyHostToDevice, stream);

	cusum.reset();

	cudaStreamSynchronize(stream);

	//3. link edges and nodes

	data_size = ctx.nEdges*sizeof(int);

	cudaMalloc(&ctx.d_edge_links, data_size);

	nThreads = 256;
	nBlocks = 1 + (ctx.nEdges - 1)/nThreads;

	kernels::graphEdgeLinkingKernel_ITk<<<nBlocks, nThreads, 0, stream>>>(ctx.d_edge_nodes, 
																	 ctx.d_edge_links, ctx.d_num_incoming_edges,
																	 ctx.nEdges);

	cudaStreamSynchronize(stream);
	
	error = cudaGetLastError();

	if(error != cudaSuccess) {
	   TRACCC_ERROR("edge linking: CUDA error: " << cudaGetErrorString(error));
	   return output_seeds;
	}

	//4. edge matching to create edge-to-edge connections

	data_size = ctx.nEdges*sizeof(unsigned char); 

	cudaMalloc(&ctx.d_num_neighbours, data_size);
	cudaMemset(ctx.d_num_neighbours, 0, data_size);

	data_size = ctx.nEdges*sizeof(int);

	cudaMalloc(&ctx.d_reIndexer, data_size);
	cudaMemset(ctx.d_reIndexer, 0xFF, data_size);	

	data_size = m_config.max_num_neighbours*ctx.nEdges * sizeof(int);
	cudaMalloc(&ctx.d_neighbours, data_size);	

	kernels::graphEdgeMatchingKernel_ITk<<<nBlocks, nThreads, 0, stream>>>(ctx.d_algo_params, ctx.d_edge_params,
											ctx.d_edge_nodes, ctx.d_num_incoming_edges, ctx.d_edge_links,
											ctx.d_num_neighbours, ctx.d_neighbours, ctx.d_reIndexer, ctx.d_counters, ctx.nEdges, m_config.max_num_neighbours);

	cudaStreamSynchronize(stream);
	cudaFree(ctx.d_edge_links);
	cudaFree(ctx.d_edge_params);	

	error = cudaGetLastError();

	if(error != cudaSuccess) {
	   TRACCC_ERROR("edge matching: CUDA error: " << cudaGetErrorString(error));
	   return output_seeds;
	}

	//5. Edge re-indexing to keep only edges involved in any connection

	kernels::edgeReIndexingKernel_ITk<<<nBlocks, nThreads, 0, stream>>>(ctx.d_reIndexer, ctx.d_counters, ctx.nEdges);

	cudaStreamSynchronize(stream);

	error = cudaGetLastError();

	if(error != cudaSuccess) {
	   TRACCC_ERROR("edge re-indexing: CUDA error: " << cudaGetErrorString(error));
	   return output_seeds;
	}

	unsigned int nStats[3];

	cudaMemcpy(&nStats[0], ctx.d_counters, 3*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	ctx.nConnections = nStats[1];
	ctx.nConnectedEdges = nStats[2];

	TRACCC_INFO("created " << ctx.nConnections << " edge links, found " << ctx.nConnectedEdges << " connected edges for seed extraction");
	if(ctx.nConnectedEdges == 0) return output_seeds;

	int nIntsPerEdge = 2 + 1 + m_config.max_num_neighbours;

	data_size = ctx.nConnectedEdges*nIntsPerEdge*sizeof(int);

	cudaMalloc(&ctx.d_output_graph, data_size);	

	nThreads = 256;
	int nEdgesPerBlock = nThreads*64;

	nBlocks = 1 + (ctx.nEdges-1)/nEdgesPerBlock;
	   
	kernels::graphCompressionKernel_ITk<<<nBlocks, nThreads, 0, stream>>>(ctx.d_sp_params, ctx.d_node_index, 
																ctx.d_edge_nodes, ctx.d_num_neighbours, ctx.d_neighbours,
																ctx.d_reIndexer, ctx.d_output_graph, nEdgesPerBlock, ctx.nEdges, m_config.max_num_neighbours);

	cudaStreamSynchronize(stream);
	
	cudaFree(ctx.d_edge_nodes);
	cudaFree(ctx.d_node_index);
	cudaFree(ctx.d_reIndexer);
	cudaFree(ctx.d_num_neighbours);
	cudaFree(ctx.d_neighbours);
	cudaFree(ctx.d_node_index);

	error = cudaGetLastError();
	if(error != cudaSuccess) {
	   TRACCC_ERROR("graph compression: CUDA error: " << cudaGetErrorString(error));
       return output_seeds;
	}
		

	//6. Find longest segments with CCA

	//7. extract seeds, longest segment first

	cudaFree(ctx.d_reducedSP);
	cudaFree(ctx.d_counters);
	cudaFree(ctx.d_output_graph);	
	cudaFree(ctx.d_algo_params);	

	TRACCC_INFO("ended!");

	return output_seeds;
	}

} //namespace traccc::cuda
