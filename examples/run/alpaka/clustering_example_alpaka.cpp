/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include "traccc/alpaka/utils/definitions.hpp"
#include "traccc/io/read_cells.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/clusterization/clusterization_algorithm.hpp"

#include "traccc/alpaka/seeding/spacepoint_binning.hpp"
#include "traccc/efficiency/seeding_performance_writer.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_spacepoints.hpp"
#include "traccc/options/common_options.hpp"
#include "traccc/options/full_tracking_input_options.hpp"
#include "traccc/options/handle_argument_errors.hpp"
#include "traccc/options/seeding_input_options.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/timer.hpp"
#include "traccc/seeding/seeding_algorithm.hpp"
#include "traccc/seeding/track_params_estimation.hpp"
#include "hitCsvReader.hpp"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#endif
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <stdio.h>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

// Alpaka
#include <alpaka/alpaka.hpp>

namespace po = boost::program_options;

struct ClusteringKernel
{
    template<typename TAcc, typename TData64, typename TData16, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
    TAcc const& acc, 
    TData64 const* const geoIDBuf,
    TData16 const* const c0Buf,
    TData16 const* const c1Buf,
    TData64* const outputBuf,
    TIdx const& numElements) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        unsigned int i = linearizedGlobalThreadIdx[0];

        // if (geoIDBuf[i] == 576461027181406464) {
            for (int x = 0; x < numElements; x++) {
                if ((geoIDBuf[x] == geoIDBuf[i]) &&
                    (-1 <= (c0Buf[x] - c0Buf[i]) && (c0Buf[x] - c0Buf[i]) <= 1) &&
                    (-1 <= (c1Buf[x] - c1Buf[i]) && (c1Buf[x] - c1Buf[i]) <= 1)) {
                    if (x == i-1 || x == i+1) {
                        // printf("c0: %u, c1: %u FOUND c0: %u, c1: %u  i: %u, x: %u\n", c0Buf[i], c1Buf[i], c0Buf[x], c1Buf[x], i, x);
                        // printf("diff in c0: %d, c1: %d, true check: %d\n", c0Buf[x] - c0Buf[i], c1Buf[x] - c1Buf[i], (-1 <= (c0Buf[x] - c0Buf[i]) && (c0Buf[x] - c0Buf[i]) <= 1));
                        outputBuf[i] = x;
                    }
                }
            }
        // }
    }
};

bool outputTest(uint16_t c0_0, uint16_t c0_1, uint16_t c1_0, uint16_t c1_1, uint64_t geoID_0, uint64_t geoID_1) {
    return (-1 <= (c0_0 - c0_1) && (c0_0 - c0_1) <= 1 && -1 <= (c1_0 - c1_1) && (c1_0 - c1_1) <= 1 && geoID_0 == geoID_1);
}

auto main(int argc, char* argv[]) -> int
{
    // Set up the program options
    po::options_description desc("Allowed options");

    // Add options
    desc.add_options()("help,h", "Give some help with the program's options");
    traccc::full_tracking_input_config full_tracking_input_cfg(desc);
    desc.add_options()("run_cpu", po::value<bool>()->default_value(false),
                    "run cpu clustering as well");
    traccc::common_options common_opts(desc);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    // Check errors
    traccc::handle_argument_errors(vm, desc);

    common_opts.read(vm);
    full_tracking_input_cfg.read(vm);
    auto run_cpu = vm["run_cpu"].as<bool>();

    traccc::hitCsvReader csvHits(common_opts.input_directory);

// Fallback for the CI with disabled sequential backend
#if defined(ALPAKA_CI) && !defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    return EXIT_SUCCESS;
#else
    // Define the index domain
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;
    using Host = alpaka::AccCpuSerial<Dim, Idx>;
    // std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << "\n"
    //         << "Using host: " << alpaka::getAccName<Host>() << std::endl;

    using AccQueueProperty = alpaka::Blocking;
    using DevQueue = alpaka::Queue<Acc, AccQueueProperty>;

    // choose between Blocking and NonBlocking
    using HostQueueProperty = alpaka::Blocking;
    using HostQueue = alpaka::Queue<Host, HostQueueProperty>;

    // Select devices
    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto const devHost = alpaka::getDevByIdx<Host>(0u);

    // Create queues
    DevQueue devQueue(devAcc);
    HostQueue hostQueue(devHost);

    uint const csvRows = sizeof(csvHits.data.geoID) / sizeof(u_int64_t);

    // Define the work division for kernels to be run on devAcc and devHost
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(1)));
    Vec const threadsPerGrid(Vec::all(static_cast<Idx>(csvRows)));
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

    WorkDiv const devWorkDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted
    );

    using Data = std::uint64_t;
    using DataChannel = std::uint16_t;
    constexpr Idx nElementsPerDim = csvRows; 

    const Vec extents(Vec::all(static_cast<Idx>(nElementsPerDim)));

    // Allocate 3 host memory buffers
    using BufHostU64 = alpaka::Buf<Host, Data, Dim, Idx>;
    using BufHostU16 = alpaka::Buf<Host, DataChannel, Dim, Idx>;

    BufHostU64 outputBuf(alpaka::allocBuf<Data, Idx>(devHost, extents));
    BufHostU64 geoIDBuf(alpaka::allocBuf<Data, Idx>(devHost, extents));
    BufHostU16 c0Buf(alpaka::allocBuf<DataChannel, Idx>(devHost, extents));
    BufHostU16 c1Buf(alpaka::allocBuf<DataChannel, Idx>(devHost, extents));

    // Initialize the host input vectors
    Data* const pOutputBuf(alpaka::getPtrNative(outputBuf));
    Data* const pGeoIDBuf(alpaka::getPtrNative(geoIDBuf));
    DataChannel* const pC0Buf(alpaka::getPtrNative(c0Buf));
    DataChannel* const pC1Buf(alpaka::getPtrNative(c1Buf));

    // Assign data 
    Idx const numElements(nElementsPerDim);
    double timeTotal = 0;
    // uint64_t tempGeoID = csvHits.data.geoID[0];
    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        for(Idx i(0); i < numElements; ++i)
        {   
            pOutputBuf[i] = i;
            pGeoIDBuf[i] = csvHits.data.geoID[i];
            pC0Buf[i] = csvHits.data.channel0[i];
            pC1Buf[i] = csvHits.data.channel1[i];
        }

        const auto endT = std::chrono::high_resolution_clock::now();
        timeTotal += std::chrono::duration<double>(endT - beginT).count();
        std::cout << "Time for assigning data to buffers, len: " << std::chrono::duration<double>(endT - beginT).count() << "s, " 
                  << std::to_string(numElements) << std::endl;
    }

    using BufAccU64 = alpaka::Buf<Acc, Data, Dim, Idx>;
    using BufAccU16 = alpaka::Buf<Acc, DataChannel, Dim, Idx>;

    BufAccU64 devOutputBuf(alpaka::allocBuf<Data, Idx>(devAcc, extents));
    BufAccU64 devGeoIDBuf(alpaka::allocBuf<Data, Idx>(devAcc, extents));
    BufAccU16 devC0Buf(alpaka::allocBuf<DataChannel, Idx>(devAcc, extents));
    BufAccU16 devC1Buf(alpaka::allocBuf<DataChannel, Idx>(devAcc, extents));
     
    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::memcpy(devQueue, devOutputBuf, outputBuf);
        alpaka::memcpy(devQueue, devGeoIDBuf, geoIDBuf);
        alpaka::memcpy(devQueue, devC0Buf, c0Buf);
        alpaka::memcpy(devQueue, devC1Buf, c1Buf);

        alpaka::wait(devQueue);
        const auto endT = std::chrono::high_resolution_clock::now();
        timeTotal += std::chrono::duration<double>(endT - beginT).count();
        std::cout << "Time for host to dev mem copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    ClusteringKernel clusteringKernel;

    auto const devClusteringKernel = alpaka::createTaskKernel<Acc>(
        devWorkDiv,
        clusteringKernel,
        alpaka::getPtrNative(devGeoIDBuf),
        alpaka::getPtrNative(devC0Buf),
        alpaka::getPtrNative(devC1Buf),
        alpaka::getPtrNative(devOutputBuf),
        numElements
    );

    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(devQueue, devClusteringKernel);
        alpaka::wait(devQueue); // wait in case we are using an asynchronous queue to time actual kernel runtime
        const auto endT = std::chrono::high_resolution_clock::now();
        timeTotal += std::chrono::duration<double>(endT - beginT).count();
        std::cout << "Time for clustering kernel execution on GPU: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::memcpy(devQueue, outputBuf, devOutputBuf);
        alpaka::memcpy(devQueue, geoIDBuf,  devGeoIDBuf);
        alpaka::memcpy(devQueue, c0Buf, devC0Buf);
        alpaka::memcpy(devQueue, c1Buf, devC1Buf);

        alpaka::wait(devQueue);
        const auto endT = std::chrono::high_resolution_clock::now();
        timeTotal += std::chrono::duration<double>(endT - beginT).count();
        std::cout << "Time for dev to host mem copy: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    // results testing 
    int numInCluster = 0;
    int clustersInGeoID = 0;
    int clustersTotal = 0;
    int geoIDTotal = 0;
    int cellsRead = 0;
    uint16_t currClusterC0[100];
    uint16_t currClusterC1[100];
    int rootIndex = 0;
    uint64_t tempGeoID = geoIDBuf[0];
    double printTime = 0;

    // cluster calc and printing
    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        printf("==========================================================\n");
        for (uint y = 0; y < numElements; y++) {
            cellsRead += 1;
            if (geoIDBuf[y] != tempGeoID) {
                clustersTotal += clustersInGeoID;
                printf("number of clusters in geoID 0x%lu: %d\n", tempGeoID, clustersInGeoID);
                printf("Total clusters found: %d\n", clustersTotal);
                printf("==========================================================\n");
                tempGeoID = geoIDBuf[y];
                clustersInGeoID = 0;
                geoIDTotal += 1;
            } 

            // if the next cell is: 
            if (outputBuf[y] == y+1) { // in the same cluster
                currClusterC0[numInCluster] = c0Buf[y];
                currClusterC1[numInCluster] = c1Buf[y];
                numInCluster++;
                outputBuf[y] = rootIndex;
            } else if (outputBuf[y] == y-1 || outputBuf[y] == y) { // a cluster by itself (cluster size of 1) or if its the end of a cluster
                currClusterC0[numInCluster] = c0Buf[y];
                currClusterC1[numInCluster] = c1Buf[y];
                numInCluster++;
                clustersInGeoID++;

                printf("cluster %d: ", clustersInGeoID);
                for (uint16_t i = 0; i < numInCluster; i++){
                    printf("(%u, %u), ", currClusterC0[i], currClusterC1[i]);
                }

                printf("\ncluser size: %d\n", numInCluster);
                printf("----------------------------------------------------------\n");

                std::fill(std::begin(currClusterC0), std::end(currClusterC0), 0); // reset cluster c0 and c1 buffers
                std::fill(std::begin(currClusterC1), std::end(currClusterC1), 0);
                
                rootIndex = outputBuf[y];
                numInCluster = 0;
            }

            
            // printf("out[%lu]: %u\n", y, outputBuf[y]);
            // printf("to test c0_0: %d, c0_1: %d, c1_0: %d, c1_1: %d\n", c0Buf[y], c0Buf[outputBuf[y]], c1Buf[y], c1Buf[outputBuf[y]]);
            // // printf("boolean test: c0: %d, c1: %d\n", -1 <= c0Buf[y] - c0Buf[outputBuf[y]] <= 1);
            // printf("correct?: %d\n", outputTest(c0Buf[y], c0Buf[outputBuf[y]], c1Buf[y], c1Buf[outputBuf[y]], geoIDBuf[y], geoIDBuf[outputBuf[y]]));
        }

        printf("Cells Read: %d\n", cellsRead);
        printf("Total clusters found: %d\n", clustersTotal);
        printf("Total unique geoIDs found: %d\n", geoIDTotal);
        const auto endT = std::chrono::high_resolution_clock::now();
        printTime += std::chrono::duration<double>(endT - beginT).count();
        std::cout << "Time for cluster printing and calc: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    printf("Time for completion %fs \nTime for completion w/o print + calc time %fs\n", timeTotal + printTime, timeTotal);

    return EXIT_SUCCESS;
#endif
}