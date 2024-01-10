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

/*
int seq_run(const traccc::full_tracking_input_config& i_cfg,
            const traccc::common_options& common_opts, bool run_cpu) {  

    // Read the surface transforms
    auto surface_transforms =
        traccc::io::read_geometry(common_opts.detector_file);

    // Read the digitization configuration file
    auto digi_cfg =
        traccc::io::read_digitization_config(i_cfg.digitization_config_file);

    // output stats
    uint64_t n_clusters = 0;
    uint64_t n_clusterRatio = 0;
    uint64_t n_clusterHits = 0;
    uint64_t n_cells = 0;
    uint64_t n_modules = 0;

    vecmem::host_memory_resource host_mr;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    vecmem::cuda::copy copy;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &host_mr};
#else
    vecmem::copy copy;
    traccc::memory_resource mr{host_mr, &host_mr};
#endif

    traccc::clusterization_algorithm ca(host_mr);

    traccc::performance::timing_info elapsedTimes;

    for (unsigned int event = common_opts.skip;
         event < common_opts.events + common_opts.skip; ++event) {
        
        traccc::io::cell_reader_output read_out_per_event(mr.host);
        traccc::clusterization_algorithm::output_type measurements_per_event;

        traccc::io::spacepoint_reader_output reader_output(mr.host);
        traccc::clusterization_algorithm::output_type cells;
        traccc::clusterization_algorithm::output_type modules;

        traccc::spacepoint_collection_types::host& spacepoints_per_event =
            reader_output.spacepoints;
        
        {  // Start measuring wall time
            traccc::performance::timer wall_t("Wall time", elapsedTimes);

            // -----------------
            // hit file reading
            // -----------------
            {
                traccc::performance::timer t("Hit reading  (cpu)",
                                             elapsedTimes);
                // Read the hits from the relevant event file
                traccc::io::read_cells(read_out_per_event, event,
                                       common_opts.input_directory,
                                       common_opts.input_data_format,
                                       &surface_transforms, &digi_cfg);
            }  // stop measuring hit reading timer


            const traccc::cell_collection_types::host& cells_per_event =
                read_out_per_event.cells;
            const traccc::cell_module_collection_types::host&
                modules_per_event = read_out_per_event.modules;
            traccc::cell_collection_types::buffer cells_buffer(
                cells_per_event.size(), mr.main);

            copy(vecmem::get_data(cells_per_event), cells_buffer);
            traccc::cell_module_collection_types::buffer modules_buffer(
                modules_per_event.size(), mr.main);
            copy(vecmem::get_data(modules_per_event), modules_buffer);

            {
                traccc::performance::timer t("Clusterization  (gpu)", elapsedTimes);
                measurements_per_event =
                    ca(cells_per_event, modules_per_event);
            }  // stop measuring clusterization cpu timer

            // {
            // // Copy the spacepoint data to the device.
            // traccc::spacepoint_collection_types::buffer
            //     spacepoints_alpaka_buffer(spacepoints_per_event.size(),
            //                               *mr.host);
            // copy(vecmem::get_data(spacepoints_per_event),
            //      spacepoints_alpaka_buffer);
            // }

            // {  // clustering binning for alpaka
            //     traccc::performance::timer t("clustering (alpaka)",
            //                                  elapsedTimes);
            //     m_spacepoint_binning(
            //         vecmem::get_data(spacepoints_alpaka_buffer));
            // }

            
            if (run_cpu) {

                // -----------------------------
                //     Clusterization (cpu)
                // -----------------------------

                {
                    traccc::performance::timer t("Clusterization  (cpu)",
                                                 elapsedTimes);
                    measurements_per_event =
                        ca(cells_per_event, modules_per_event);
                }  // stop measuring clusterization cpu timer
            }

            // printf(measurements_per_event);
            // printf("Measurements per event: \n %s \n"(traccc::cell_collection_types)(&measurements_per_event).c_str());
            // // std::cout << measurements_per_event << std::endl; 

        }


    }

    // if (common_opts.check_performance) {
    //     sd_performance_writer.finalize();
    // }
    // std::cout << "==> Statistics ... " << std::endl;
    // std::cout << "- read    " << n_clusters << " clusters from "
    //           << n_modules << " modules" << std::endl;
    // std::cout << "- created (cpu)  " << n_cells << " cells" << std::endl;
    // std::cout << "- created (alpaka) " << n_clusterHits << " cluster hits"
    //           << std::endl;
    // std::cout << "==>Elapsed times...\n" << elapsedTimes << std::endl;

    return 0;
}
*/

// int main(int argc, char* argv[]) {


//     // Set up the program options
//     po::options_description desc("Allowed options");

//     // Add options
//     desc.add_options()("help,h", "Give some help with the program's options");
//     traccc::full_tracking_input_config full_tracking_input_cfg(desc);
//     desc.add_options()("run_cpu", po::value<bool>()->default_value(false),
//                     "run cpu clustering as well");
//     traccc::common_options common_opts(desc);

//     po::variables_map vm;
//     po::store(po::parse_command_line(argc, argv, desc), vm);

//     // Check errors
//     traccc::handle_argument_errors(vm, desc);

//     common_opts.read(vm);
//     full_tracking_input_cfg.read(vm);
//     auto run_cpu = vm["run_cpu"].as<bool>();

//     traccc::hitCsvReader csvHits(common_opts.input_directory); //"/home/wthompson/Work/traccc/data/tml_full/ttbar_mu20/event000000000-cells.csv"

//     for (int i = 0; i < 10; i++) {
//        std::cout << csvHits.data.geoID[i] << "," << csvHits.data.channel0[i] << "," << csvHits.data.channel1[i]  << std::endl;
//     }

//     std::cout << "Running " << argv[0] << " "
//               << full_tracking_input_cfg.detector_file << " "
//               << common_opts.input_directory << " " << common_opts.events
//               << std::endl;

//     // return seq_run(full_tracking_input_cfg, common_opts, run_cpu);
// }


struct ClusteringKernel
{
    template<typename TAcc, typename TData>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, 
    TData const* const geoIDBuf,
    TData const* const c0Buf,
    TData const* const c1Buf
    ) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx = alpaka::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        unsigned int i = linearizedGlobalThreadIdx[0];
        for(int j = 0; j < 10; ++j)
        //{
            printf("Thread ID %u, GeoID %u, Channel0 %u, Channel1 %u\n", linearizedGlobalThreadIdx[0], geoIDBuf[i], c0Buf[i], c1Buf[i]);
        //}

        // printf("Thread ID %d, %d\n", linearizedGlobalThreadIdx, geoIDBuf[0]);
    }
};

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

    for (int i = 0; i < 10; i++) {
       std::cout << csvHits.data.geoID[i] << "," << csvHits.data.channel0[i] << "," << csvHits.data.channel1[i]  << std::endl;
    }

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

    // Define the work division for kernels to be run on devAcc and devHost
    using Vec = alpaka::Vec<Dim, Idx>;
    Vec const elementsPerThread(Vec::all(static_cast<Idx>(10)));
    Vec const threadsPerGrid(Vec::all(static_cast<Idx>(100)));
    using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
    WorkDiv const devWorkDiv = alpaka::getValidWorkDiv<Acc>(
        devAcc,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted
    );

    WorkDiv const hostWorkDiv = alpaka::getValidWorkDiv<Host>(
        devHost,
        threadsPerGrid,
        elementsPerThread,
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted
    );

    using Data = std::uint32_t;
    using DataChannel = std::uint16_t;
    constexpr Idx nElementsPerDim = 100; 

    const Vec extents(Vec::all(static_cast<Idx>(nElementsPerDim)));
    
    // std::array<Data, nElementsPerDim * nElementsPerDim * nElementsPerDim> plainBuffer;
    // auto hostViewPlainPtr = alpaka::createView(devHost, plainBuffer.data(), extents);

    // Allocate 3 host memory buffers
    using BufHost = alpaka::Buf<Host, Data, Dim, Idx>;
    BufHost geoIDBuf(alpaka::allocBuf<Data, Idx>(devHost, extents));
    BufHost c0Buf(alpaka::allocBuf<Data, Idx>(devHost, extents));
    BufHost c1Buf(alpaka::allocBuf<Data, Idx>(devHost, extents));

    // Initialize the host input vectors
    Data* const pGeoIDBuf(alpaka::getPtrNative(geoIDBuf));
    Data* const pC0Buf(alpaka::getPtrNative(c0Buf));
    Data* const pC1Buf(alpaka::getPtrNative(c1Buf));

    // Assign data 
    Idx const numElements(100);

    for(Idx i(0); i < numElements; ++i)
    {
        pGeoIDBuf[i] = 100+i; // csvHits.data.geoID[i];
        pC0Buf[i] = 75+i; // csvHits.data.channel0[i];
        pC1Buf[i] = 25+i; // csvHits.data.channel1[i];
    }

    using BufAcc = alpaka::Buf<Acc, Data, Dim, Idx>;
    BufAcc devGeoIDBuf(alpaka::allocBuf<Data, Idx>(devAcc, extents));
    BufAcc devC0Buf(alpaka::allocBuf<Data, Idx>(devAcc, extents));
    BufAcc devC1Buf(alpaka::allocBuf<Data, Idx>(devAcc, extents));


    alpaka::memcpy(devQueue, devGeoIDBuf, geoIDBuf);
    alpaka::memcpy(devQueue, devC0Buf, c0Buf);
    alpaka::memcpy(devQueue, devC1Buf, c1Buf);

    alpaka::wait(devQueue);

    // for(Idx i(0); i < numElements; ++i)
    // {
    //     devGeoIDBuf[i] = 50; // csvHits.data.geoID[i];
    //     devC0Buf[i] = csvHits.data.channel0[i];
    //     devC1Buf[i] = csvHits.data.channel1[i];
    // }
    // Depending on the accelerator, the allocation function may introduce
    // padding between rows/planes of multidimensional memory allocations.
    // Therefore the pitch (distance between consecutive rows/planes) may be
    // greater than the space required for the data.
    // Idx const deviceBuffer1Pitch(alpaka::getPitchBytes<2u>(deviceBuffer1) / sizeof(Data));
    // Idx const deviceBuffer2Pitch(alpaka::getPitchBytes<2u>(deviceBuffer2) / sizeof(Data));
    // Idx const hostBuffer1Pitch(alpaka::getPitchBytes<2u>(hostBuffer) / sizeof(Data));
    // Idx const hostViewPlainPtrPitch(alpaka::getPitchBytes<2u>(hostViewPlainPtr) / sizeof(Data));

    // Test device Buffer
    //
    // This kernel tests if the copy operations
    // were successful. In the case something
    // went wrong an assert will fail.
    // Data const* const pDeviceBuffer1 = alpaka::getPtrNative(devGeoIDBuf);
    // Data const* const pDeviceBuffer2 = alpaka::getPtrNative(deviceBuffer2);

    // Data* const pHostViewPlainPtr = alpaka::getPtrNative(hostViewPlainPtr);

    ClusteringKernel clusteringKernel;

    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::exec<Host>(
            hostQueue,
            hostWorkDiv,
            clusteringKernel,
            alpaka::getPtrNative(geoIDBuf),
            alpaka::getPtrNative(c0Buf),
            alpaka::getPtrNative(c1Buf)
        ); 
        alpaka::wait(hostQueue); // wait in case we are using an asynchronous queue to time actual kernel runtime
        const auto endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution on CPU: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    {
        const auto beginT = std::chrono::high_resolution_clock::now();
        alpaka::exec<Acc>(
            devQueue,
            devWorkDiv,
            clusteringKernel,
            alpaka::getPtrNative(devGeoIDBuf),
            alpaka::getPtrNative(devC0Buf),
            alpaka::getPtrNative(devC1Buf)
        );
        alpaka::wait(devQueue); // wait in case we are using an asynchronous queue to time actual kernel runtime
        const auto endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for kernel execution on GPU: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }

    return EXIT_SUCCESS;
#endif
}