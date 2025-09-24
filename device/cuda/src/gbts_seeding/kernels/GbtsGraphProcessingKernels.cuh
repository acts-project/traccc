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
#include <cuda_fp16.h>

namespace traccc::cuda::kernels {

//currently 80 bytes -> 5 v4 loads/stores
struct __align__(16) edgeState {
		
	__device__ inline void initialize(const float4& node1_params, const float4& node2_params);

	__device__ inline float& m_Cx(const int i, const int j) {return Cx[i + j + 1*(i == 0)*(j == 0)];}
	__device__ inline float& m_Cy(const int i, const int j) {return Cy[i + j];}
	__device__ inline const float& m_Cx(const int i, const int j) const {return Cx[i + j + 1*(i == 0)*(j == 0)];}
	__device__ inline const float& m_Cy(const int i, const int j) const {return Cy[i + j];}

	float m_X[3], m_Y[2];
	float m_c, m_s, m_refX, m_refY;

	int m_J : 31;

	bool m_head_node_type : 1;

	unsigned int m_mini_idx : 27;
	unsigned int m_length : 5;
	int m_edge_idx;

	//upper triangle of the Cov matrix for the parabola in the x,y plane since symetry gives the rest
	float Cx[5]; //(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
	//Cov matrix for the linear fit of eta and z
	float Cy[3]; //(0,0), (0,1), (1,1)
};

struct Tracklet {
	unsigned int nodes[traccc::device::gbts_consts::max_cca_iter+1];
	int size;
};

/** @brief Performs one iteration of the CCA over the graph to calculate potential seed length
*
*  This also constructs a level view over the graph edges for use in seed extraction
*
*  @param[in] d_output_graph see comments in device_context.h 
*  @param[in] d_levels is the maximum seed length originating at each edge
*  @param[in/out] d_active_edges stores the edge_idx for edges that have not reached their level
*  @param[out] d_level_views / d_level_boundaires together they construct a view over the edges by level for use in seed extraction
*  @param[internal] d_counters[5 and 6] used for counting the last block and for the number of active edges 
*/
__global__ static void CCA_IterationKernel(const int* d_output_graph, char* d_levels, int* d_active_edges, int* d_level_views,
                                           int* d_level_boundaries, unsigned int* d_counters, int iter, int nEdges, int max_num_neighbours) {

	__shared__ int nEdgesLeft;
	int edge_size = 2 + 1 + max_num_neighbours;

	int toggle     = iter%2;
	int levelLoad  = toggle*nEdges;
	int levelStore = (1-toggle)*nEdges;

	if(threadIdx.x == 0) {
		nEdgesLeft = d_counters[3+toggle];//from the previous iteration
	}
	__syncthreads();

	for(int globalIdx = threadIdx.x + blockIdx.x*blockDim.x; globalIdx < nEdgesLeft; globalIdx += blockDim.x*gridDim.x) {
		
		int edgeIdx = iter == 0 ? globalIdx : d_active_edges[globalIdx];

		int edge_pos = edge_size*edgeIdx;
		
		int nNei = d_output_graph[edge_pos + traccc::device::gbts_consts::nNei];
		
		char next_level = d_levels[levelLoad + edgeIdx];

		bool localChange = false;
		for(int nIdx = 0; nIdx < nNei; nIdx++) {//loop over neighbouring edges
			
			int nextEdgeIdx = d_output_graph[edge_pos + traccc::device::gbts_consts::nei_start + nIdx];
			char forward_level = d_levels[levelLoad + nextEdgeIdx];

			if(next_level == forward_level) {
				next_level = forward_level + 1;
				localChange = true;
				break;
			}
		}
		// add all remianing edges to level_views on the last iteration
		if(localChange && iter < traccc::device::gbts_consts::max_cca_iter - 1) {
			int edgesLeftPlace = atomicAdd(&d_counters[4-toggle], 1); //nChanges
			d_active_edges[edgesLeftPlace] = edgeIdx;//for the next iteration
		}
		else {
			d_level_views[atomicAdd(&d_counters[6], 1)] = edgeIdx;
		}
		d_levels[levelStore + edgeIdx] = next_level; //store new level and ensure all final levels are on both sides of the array
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		if(atomicAdd(&d_counters[5], 1) == gridDim.x-1) {//this is the last block
			d_level_boundaries[iter] = nEdgesLeft;
			d_counters[3+toggle] = 0;
			d_counters[5] = 0;
		}
	}
}


/** @brief initialize the Kalman filter for this new edgeState from the starting edge (2 nodes)
*
*  @param[in] node1_params / node2_params the nodes of the starting edge. Node 1 is the inner node and we filter outside in 
*/
__device__ inline void edgeState::initialize(const float4& node1_params, const float4& node2_params) { //x, y, z,type

	m_J = 0;
	m_length = 1;
	m_head_node_type = (node1_params.w < 0);
	//n2->n1

	float dx = node1_params.x-node2_params.x;
	float dy = node1_params.y-node2_params.y;
	float L  = sqrtf(dx*dx + dy*dy);

	float r1 = sqrtf(node1_params.x*node1_params.x + node1_params.y*node1_params.y);
	float r2 = sqrtf(node2_params.x*node2_params.x + node2_params.y*node2_params.y);

	m_s = dy/L;
	m_c = dx/L;

	//transform for extrapolation and update
	// x' =  x*m_c + y*m_s
	// y' = -x*m_s + y*m_c

	m_refY = r2;
	m_refX = node2_params.x*m_c + node2_params.y*m_s;

	//X-state: y, dy/dx, d2y/dx2

	m_X[0] = -node2_params.x*m_s + node2_params.y*m_c;
	m_X[1] = 0.0f;
	m_X[2] = 0.0f;

	//Y-state: z, dz/dr

	m_Y[0] = node2_params.z;
	m_Y[1] = (node1_params.z - node2_params.z)/(r1 - r2);

	memset(&m_Cx(0,0), 0, sizeof(Cx));
	memset(&m_Cy(0,0), 0, sizeof(Cy));

	m_Cx(0,0) = 0.25f;
	m_Cx(1,1) = 0.001f;
	m_Cx(2,2) = 0.001f;

	m_Cy(0,0) = 1.5f;
	m_Cy(1,1) = 0.001f;

}

/** Attempts to update the edgeState to include node1
*
*  This is a Kalamn filter update fitting to strait line in z,r and a parabola in x',y' with the transformation defined by the first edge when the state is initialized  
*  It's main output is the m_J seed quality used for disambiguation
*  Seed extraction goes outside in
*  
*  @param[out] new_ts output edgeState for the updated seed including node1
*  @param[in] ts input edgeState is const because it will be re used for each of its head edge's connections
*  @param[in] node1_params params of the inner node of the new edge to be added to the seed
*/
inline __device__ bool update(edgeState* new_ts, const edgeState* ts, const float4& node1_params) { //params are x, y, z, type

	const float sigma_t = 0.0003f;
	const float sigma_w = .00009f;

	const float sigmaMS = 0.016f;

	const float sigma_x = 0.25f;//was 0.22
	const float sigma_y = 2.5f;//was 1.7

	const float weight_x = 0.5f;
	const float weight_y = 0.5f;

	const float maxDChi2_x = 60.0f;//was 35.0;
	const float maxDChi2_y = 60.0f;//was 31.0;

	const float add_hit = 14.0f;
	//m_J is stored in 30 + sign bits so max qual = INT_MAX/2 = add_hit*max_length*qual_scale
	const float qual_scale = 0.5*static_cast<float>(INT_MAX)/static_cast<float>(add_hit*traccc::device::gbts_consts::max_cca_iter) - 1;

	//add ms.
	float m_Cx22 = ts->m_Cx(2,2) + sigma_w*sigma_w;
	float m_Cx11 = ts->m_Cx(1,1) + sigma_t*sigma_t;

	float t2 = node1_params.w != -1 ? 1 + ts->m_Y[1]*ts->m_Y[1] : 1 + 1/(ts->m_Y[1]*ts->m_Y[1]);

	float s1 = sigmaMS*t2;
	float s2 = s1*s1;

	s2 *= sqrtf(t2);

	float m_Cy11 = ts->m_Cy(1,1) + s2;

	//extrapolation

	//float refX, refY;
	float mx, my;

	float r = sqrtf(node1_params.x*node1_params.x + node1_params.y*node1_params.y);

	//using new_ts as register storage where possible   
	new_ts->m_refX = node1_params.x*ts->m_c + node1_params.y*ts->m_s;
	mx   = -node1_params.x*ts->m_s + node1_params.y*ts->m_c;//measured X[0]
	new_ts->m_refY = r;
	my   = node1_params.z;//measured Y[0]
		
	float A = new_ts->m_refX - ts->m_refX;
	float B = (0.5f)*A*A;
	float dr = new_ts->m_refY - ts->m_refY;

	new_ts->m_X[0] = ts->m_X[0] + ts->m_X[1]*A + ts->m_X[2]*B;
	new_ts->m_X[1] = ts->m_X[1] + ts->m_X[2]*A;
	new_ts->m_X[2] = ts->m_X[2];

	new_ts->m_Cx(0,0) = ts->m_Cx(0,0) + 2*ts->m_Cx(0,1)*A + 2*ts->m_Cx(0,2)*B + A*m_Cx11*A + 2*A*ts->m_Cx(1,2)*B + B*m_Cx22*B;
	new_ts->m_Cx(0,1) = ts->m_Cx(0,1) + m_Cx11*A + ts->m_Cx(1,2)*B + ts->m_Cx(0,2)*A + A*A*ts->m_Cx(1,2)  + A*m_Cx22*B;
	new_ts->m_Cx(0,2) = ts->m_Cx(0,2) + ts->m_Cx(1,2)*A + m_Cx22*B;
		
	new_ts->m_Cx(1,1) = m_Cx11 + 2*A*ts->m_Cx(1,2) + A*m_Cx22*A;
	new_ts->m_Cx(1,2) = ts->m_Cx(1,2) + m_Cx22*A;

	new_ts->m_Cx(2,2) = m_Cx22;

	new_ts->m_Y[0] = ts->m_Y[0] + ts->m_Y[1]*dr;
	new_ts->m_Y[1] = ts->m_Y[1];

	new_ts->m_Cy(0,0) = ts->m_Cy(0,0) + 2*ts->m_Cy(0,1)*dr + dr*m_Cy11*dr;
	new_ts->m_Cy(0,1) = ts->m_Cy(0,1) + dr*m_Cy11;
	new_ts->m_Cy(1,1) = m_Cy11;

	//chi2 test
	float resid_x = mx - new_ts->m_X[0];
	float resid_y = my - new_ts->m_Y[0];

	float sigma_rz = 0;

	if(!ts->m_head_node_type) {//barrel TO-DO: split into barrel Pixel and barrel SCT
		sigma_rz = sigma_y*sigma_y;
	}
	else {
		sigma_rz = sigma_y*ts->m_Y[1];
		sigma_rz = sigma_rz*sigma_rz;
	}

	float Dx = 1/(new_ts->m_Cx(0,0) + sigma_x*sigma_x);

	float Dy = 1/(new_ts->m_Cy(0,0) + sigma_rz);

	float dchi2_x = resid_x*resid_x*Dx;
	float dchi2_y = resid_y*resid_y*Dy;

	if(dchi2_x > maxDChi2_x || dchi2_y > maxDChi2_y) {
		return false;
	}

	//state update
	new_ts->m_J = ts->m_J + static_cast<int>((add_hit - dchi2_x*weight_x - dchi2_y*weight_y)*qual_scale);
	new_ts->m_length = ts->m_length+1;

	for(int i=0;i<3;i++) new_ts->m_X[i] += Dx*new_ts->m_Cx(0,i)*resid_x;
	for(int i=0;i<2;i++) new_ts->m_Y[i] += Dx*new_ts->m_Cy(0,i)*resid_y;

	new_ts->m_Cx(2,2) -= Dx*new_ts->m_Cx(0,2)*new_ts->m_Cx(0,2);
	new_ts->m_Cx(1,2) -= Dx*new_ts->m_Cx(0,1)*new_ts->m_Cx(0,2);
	new_ts->m_Cx(1,1) -= Dx*new_ts->m_Cx(0,1)*new_ts->m_Cx(0,1);
	new_ts->m_Cx(0,2) -= Dx*new_ts->m_Cx(0,0)*new_ts->m_Cx(0,2);
	new_ts->m_Cx(0,1) -= Dx*new_ts->m_Cx(0,0)*new_ts->m_Cx(0,1);
	new_ts->m_Cx(0,0) -= Dx*new_ts->m_Cx(0,0)*new_ts->m_Cx(0,0);

	new_ts->m_Cy(1,1) -= Dx*new_ts->m_Cy(0,1)*new_ts->m_Cy(0,1);
	new_ts->m_Cy(0,1) -= Dx*new_ts->m_Cy(0,0)*new_ts->m_Cy(0,1);
	new_ts->m_Cy(0,0) -= Dx*new_ts->m_Cy(0,0)*new_ts->m_Cy(0,0);

	new_ts->m_c = ts->m_c;
	new_ts->m_s = ts->m_s;
	new_ts->m_head_node_type = (node1_params.w < 0);

	return true;
}

/** @brief Performs seed disambiguation through seeds biding to use edges with seed quality
*
*  @param[in] m_J is the quality metric output by the Kalman filter
*  @param[in] mini_idx the index of final mini_state, backtracking from this gives the seeds path (edges->nodes)
*  @param[in] d_mini_states stores the path each seed took through the graph in reverse order
*  @param[in] prop_idx the index of this new seeds proposition in d_seed_proposals
*  @param[in/out] d_edge_bids is [int_m_J, prop_idx] so that atomicMax will swap it out with higher quality bids. The index is then used to flag the replaced seed as maybe fake
*  @param[out] d_seed_proposals stores the information needed to construct an output Tracklet for this seed
*  @param[out] d_seed_ambiguity here is 0 if the seed is the highest quality seed using all of its edges and -1 otherwise
*/
inline __device__ void add_seed_proposal(const int m_J, const int mini_idx, const unsigned int prop_idx, char* d_seed_ambiguity, int2* d_seed_proposals,
                                         unsigned long long int* d_edge_bids, const int2* d_mini_states) {
	
	//new seed bids for its edges
	d_seed_proposals[prop_idx] = make_int2(m_J, mini_idx);
	d_seed_ambiguity[prop_idx] = 0;
	__threadfence(); //ensure above proposal info is written before biding

	unsigned long long int seed_bid = (static_cast<unsigned long long int>(m_J) << 32) | (static_cast<unsigned long long int>(prop_idx));

	int2 mini_state;
	for(int next_mini = mini_idx; next_mini >= 0;) {
		mini_state = d_mini_states[next_mini];
		
		unsigned long long int competing_offer = atomicMax(&d_edge_bids[mini_state.x], seed_bid);
		if(competing_offer > seed_bid) {d_seed_ambiguity[prop_idx] = -1;}
		else if(competing_offer != 0) {d_seed_ambiguity[competing_offer & 0xFFFFFFFFLL] = -1;} //default bids are 0 so no need to replace
		
		next_mini = mini_state.y;
	}
}

/** @brief This kerenel extracts the seeds from the graph starting with sets of edges with simmlar levels
*
*  We start with higher levels so that the best seeds are found first and pruned from the graph
*  The last block to finsh perfoms disambiguation through iteritive biding and fills d_seeds with the winning seeds with the nodes in inside-out order
*
*  @param[in] view_min/view_max the range of input edges in the d_level_views to start forming seeds from 
*  @param[in] d_level_views view on the edges of d_output_graph calculated by the CCA
*  @param[in/out] d_levels the level of each edge by edge, -1 signifes an edge that is allready used in seed found from a previous iteration and so has been removed from the graph 
*  @param[in] d_sp_params x,y,z,cluster-width for all nodes. Here cluster width denotes if a sp is in the barrel (cw != -1 => barrel)
*  @param[in] d_output_graph stores the nodes, number of neighbours and self-referential neighbour index.
*  @param[in] minLevel is length, in edges, of the smallest accepable seed (and is 3 by default so 4 space points)
*  @param[internal] d_counters[7,8 and 10] are used to track the length of d_mini_state, d_seed_proposals and number of finished blocks
*  @param[internal] d_seed_ambiguity, d_edge_bids used in seed diambiguation for details see add_seed_proposal  
*  @param[internal] d_mini_states stores the path each seed took through the graph in reverse order (inside out)
*  @param[internal] d_seed_proposals stores the infomation needed to construct an output Tracklet for this seed
*  @param[out] d_seeds stores the output seeds after disambiguation. This is what gets transfered back to CPU and is the final output of these kernels
*  @param[out] d_counters[9] number of seeds in d_seeds
*/
__global__ void seed_extracting_kernel(int view_min, int view_max, int* d_level_views, char* d_levels, float4* d_sp_params, int* d_output_graph, 
                     int2* d_mini_states, edgeState* d_state_store, unsigned long long int* d_edge_bids, char* d_seed_ambiguity, int2* d_seed_proposals, Tracklet* d_seeds,
                     unsigned int* d_counters, int minLevel, int nMaxMini, int nMaxProps, int nMaxStateStorePerBlock, int nMaxSeeds, int max_num_neighbours) {

	__shared__ int block_start;

	__shared__ int total_live_states;
	__shared__ int nStates;
	__shared__ int nSharedSpace;

	__shared__ edgeState current_states[traccc::device::gbts_consts::shared_state_buffer_size];

	int edge_size = 2 + 1 + max_num_neighbours;
	//TO-DO? each block to take the same distribution of levels
	if(threadIdx.x == 0) {
		total_live_states = 0;
		
		int total_nStates = view_max - view_min;
		nStates = 1+(total_nStates-1)/gridDim.x;
			
		block_start = view_min + nStates*blockIdx.x;
		if(block_start >= view_max) nStates = 0;
		else if(block_start + nStates >= view_max) nStates = view_max - block_start;
	}
	__syncthreads();
	//TO-DO? allow for intialization in global
	//assign root edges to blocks and populate shared with inital states
	//must have less input than shared space
	for(int root_edge_idx = threadIdx.x; root_edge_idx<nStates; root_edge_idx+=blockDim.x) {
		
		int edge_idx = d_level_views[block_start + root_edge_idx];
		char level = d_levels[edge_idx];
		if(level == -1) continue;

		int edge_pos = edge_size*edge_idx;

		float4 node1_params = d_sp_params[d_output_graph[edge_pos + traccc::device::gbts_consts::node1]];
		float4 node2_params = d_sp_params[d_output_graph[edge_pos + traccc::device::gbts_consts::node2]];

		int root_idx = atomicAdd(&total_live_states, 1);
		current_states[root_idx].initialize(node1_params, node2_params);
		current_states[root_idx].m_edge_idx = edge_idx;
		
		int mini_idx = atomicAdd(&d_counters[7], 1);
		d_mini_states[mini_idx] = make_int2(edge_idx, -1); //prev mini -1 for roots with no prev
		current_states[root_idx].m_mini_idx = mini_idx;
		
	}
	__syncthreads();
	if(threadIdx.x == 0) nStates = total_live_states; //update after removed edges are exculded
		
	edgeState state;
	edgeState new_state;

	__syncthreads();
	while(total_live_states>0) {
		// propogate to next level 
		bool has_state = false;
		if(threadIdx.x<nStates) {state = current_states[nStates-1-threadIdx.x]; has_state = true;} 
		else if(threadIdx.x<total_live_states) {state = d_state_store[total_live_states-1-threadIdx.x+nMaxStateStorePerBlock*blockIdx.x]; has_state = true;}
		__syncthreads();
		if(threadIdx.x == 0) { //update state counts
			total_live_states = (total_live_states < blockDim.x) ? 0 : total_live_states - blockDim.x;
			
			nStates = (nStates < blockDim.x) ? 0 : nStates - blockDim.x;
			nSharedSpace = total_live_states - nStates; //total state count when shared memory is filled - max shared states
		}
		__syncthreads();
		if(has_state) {

			int edge_idx = state.m_edge_idx;
			
			int edge_pos = edge_idx*edge_size;
			
			int nNei = d_output_graph[edge_pos + traccc::device::gbts_consts::nNei];
			
			char edge_level = d_levels[edge_idx];
			
			bool no_updates = true;
			
			for(unsigned char nei = 0;nei<nNei;nei++) {
				int nei_idx  = d_output_graph[edge_pos + traccc::device::gbts_consts::nei_start + nei];
				
				char nei_level = d_levels[nei_idx];
				if(edge_level - 1 != nei_level) continue;
				
				float4 node1_params = d_sp_params[d_output_graph[edge_size*nei_idx + traccc::device::gbts_consts::node1]];
				bool success = update(&new_state, &state, node1_params);
				
				if(!success) continue;
				no_updates = false;
				
				new_state.m_edge_idx = nei_idx;
				new_state.m_mini_idx = atomicAdd(&d_counters[7], 1);
				
				if(new_state.m_mini_idx < nMaxMini) {
					d_mini_states[new_state.m_mini_idx] = make_int2(nei_idx, state.m_mini_idx);
						
					if(d_output_graph[edge_size*nei_idx + traccc::device::gbts_consts::nNei] == 0) { //no neighbours so will fail next round anyway so save shared
						if(new_state.m_length >= minLevel) {
							int prop_idx = atomicAdd(&d_counters[8], 1);
							if(prop_idx < nMaxProps) add_seed_proposal(new_state.m_J, new_state.m_mini_idx, prop_idx, d_seed_ambiguity, d_seed_proposals, d_edge_bids, d_mini_states);
						}
					}
					else {
						int stateStoreIdx = atomicAdd(&total_live_states, 1) - traccc::device::gbts_consts::shared_state_buffer_size; 
						if(stateStoreIdx<nSharedSpace) {current_states[atomicAdd(&nStates, 1)] = new_state;}
						else { //TO-DO? make state_store shared between blocks
							if(stateStoreIdx<nMaxStateStorePerBlock) d_state_store[stateStoreIdx+nMaxStateStorePerBlock*blockIdx.x] = new_state;
							else d_counters[10] = stateStoreIdx;
						}
					}
				}
			}
			if(no_updates) {
				if(state.m_length >= minLevel) {
					int prop_idx = atomicAdd(&d_counters[8], 1);
					if(prop_idx < nMaxProps ) add_seed_proposal(state.m_J, state.m_mini_idx, prop_idx, d_seed_ambiguity, d_seed_proposals, d_edge_bids, d_mini_states);
				}
			}
		}
		__syncthreads(); //wait for current_states to repopulate
	}
	__syncthreads();
	//move remianing seed props to seeds after all tracking for this set is done //seperate kernel?
	if(threadIdx.x == 0) nStates = atomicAdd(&d_counters[11], 1);
	__syncthreads();
	if(nStates != gridDim.x-1) return;
	unsigned int nProps = d_counters[8];
	__syncthreads();

	//reset for next launch
	if(threadIdx.x == 0) {
		//exit if any overflows have occured
		if(nProps > nMaxProps || d_counters[7] > nMaxMini || d_counters[10] != 0) nStates = 0;
		else nStates = 1;
		d_counters[11] = 0;
		d_counters[10] = 0;
		d_counters[7]  = 0;
		d_counters[8]  = 0;
	}
	__syncthreads();
	if(nProps == 0 || nStates == 0) return; 
		
	for(int round=0; round<5 && nStates > 0 ;round++) { //re-check maybe seeds that don't clash with a definte seed
		if(threadIdx.x == 0) nStates = 0;
		__syncthreads(); // fit maybe seeds into unused spaces  
		for(int prop_idx = threadIdx.x; prop_idx<nProps; prop_idx+=blockDim.x) {
			
			char ambiguity = d_seed_ambiguity[prop_idx];
			if(ambiguity == 0 || ambiguity == -2)  continue; //is not ambiguous  
			
			int2 prop = d_seed_proposals[prop_idx];
			
			bool isgood = true;

			int2 mini_state;
			for(int next_mini = prop.y; next_mini >= 0;) {
				mini_state = d_mini_states[next_mini];
				next_mini = mini_state.y;
				
				unsigned long long int best_bid = d_edge_bids[mini_state.x];
				if(best_bid == 0) continue; //already reset
				
				if(d_seed_ambiguity[best_bid & 0xFFFFFFFFLL] == 0) {isgood = false; break;} //clashes with definate seed
				d_edge_bids[mini_state.x] = 0; //reset edge bid from (possibly) fake seed   
			}
			if(isgood) {d_seed_ambiguity[prop_idx] = 1; atomicAdd(&nStates, 1);} //flag as maybe seed 
			else d_seed_ambiguity[prop_idx] = -2; //definate fake
		}
		__syncthreads();
		for(int prop_idx = threadIdx.x; prop_idx<nProps; prop_idx+=blockDim.x) {    
			if(d_seed_ambiguity[prop_idx] != 1) continue;

			int2 prop = d_seed_proposals[prop_idx];
			
			add_seed_proposal(prop.x, prop.y, prop_idx, d_seed_ambiguity, d_seed_proposals, d_edge_bids, d_mini_states); //reset and re bid
		}
		__syncthreads();            
	}
	__syncthreads();
	for(int prop_idx = threadIdx.x; prop_idx<nProps; prop_idx+=blockDim.x) {
		if(d_seed_ambiguity[prop_idx] != 0) continue;           
		int2 prop = d_seed_proposals[prop_idx];     
		
		unsigned int seed_idx = atomicAdd(&d_counters[9], 1);
		if(seed_idx > nMaxSeeds) break;

		//add good seed to output
		int2 mini_state;
		int length = 0;
		for(int next_mini = prop.y; next_mini >= 0; length++) {
			
			mini_state = d_mini_states[next_mini];
			next_mini = mini_state.y;
			
			d_seeds[seed_idx].nodes[length] = d_output_graph[mini_state.x*edge_size + traccc::device::gbts_consts::node1]; 
			d_levels[mini_state.x] = -1; //remove edge from graph
		}
		d_seeds[seed_idx].nodes[length] = d_output_graph[mini_state.x*edge_size + traccc::device::gbts_consts::node2];
		d_seeds[seed_idx].size = ++length;
	}
	__syncthreads();

}

void __global__ gbts_seed_conversion_kernel(Tracklet* d_seeds, edm::seed_collection::view output_seeds, int nSeeds) {

	edm::seed_collection::device seeds_device(output_seeds);
	for(int tracklet = threadIdx.x + blockIdx.x*blockDim.x; tracklet < nSeeds; tracklet += blockDim.x*gridDim.x) {
		int length = d_seeds[tracklet].size;
		//sample begining, middle, end sp from tracklet for now
		seeds_device.push_back({d_seeds[tracklet].nodes[0], d_seeds[tracklet].nodes[length/2], d_seeds[tracklet].nodes[length-1]});
	}
}

} //namespace kernels
