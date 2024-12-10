/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <chrono>
#include <fstream>
#include <iostream>

#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>
#include <covfie/cuda/backend/primitive/cuda_texture.hpp>
#include <covfie/cuda/error_check.hpp>

#include "bitmap.hpp"

using cpu_field_t = covfie::field<
    covfie::backend::affine<covfie::backend::linear<covfie::backend::strided<
        covfie::vector::size3,
        covfie::backend::array<covfie::vector::float3>>>>>;

using cuda_field_t = covfie::field<covfie::backend::affine<
    covfie::backend::
        cuda_texture<covfie::vector::float3, covfie::vector::float3>>>;

void parse_opts(
    int argc, char * argv[], boost::program_options::variables_map & vm
)
{
    boost::program_options::options_description opts("general options");

    opts.add_options()("help", "produce help message")(
        "input,i",
        boost::program_options::value<std::string>()->required(),
        "input vector field to read"
    )("output,o",
      boost::program_options::value<std::string>()->required(),
      "output bitmap image to write"
    )("height,h",
      boost::program_options::value<unsigned int>()->default_value(1024),
      "height of the output image"
    )("width,w",
      boost::program_options::value<unsigned int>()->default_value(1024),
      "width of the output image"
    )("z",
      boost::program_options::value<float>()->default_value(0.f),
      "z-index of the vector field slice");

    boost::program_options::parsed_options parsed =
        boost::program_options::command_line_parser(argc, argv)
            .options(opts)
            .run();

    boost::program_options::store(parsed, vm);

    if (vm.count("help")) {
        std::cout << opts << std::endl;
        std::exit(0);
    }

    try {
        boost::program_options::notify(vm);
    } catch (boost::program_options::required_option & e) {
        BOOST_LOG_TRIVIAL(fatal) << e.what();
        std::exit(1);
    }
}

template <typename field_t>
__global__ void render(
    typename field_t::view_t vf,
    unsigned char * out,
    unsigned int width,
    unsigned int height,
    float z
)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < width && y < height) {
        float fx = x / static_cast<float>(width);
        float fy = y / static_cast<float>(height);

        typename field_t::output_t p =
            vf.at(fx * 20000.f - 10000.f, fy * 20000.f - 10000.f, z);
        out[height * x + y] = static_cast<unsigned char>(std::lround(
            255.f * fmin(
                        ::sqrtf(
                            ::powf(p[0] / 0.000299792458f, 2.f) +
                            ::powf(p[1] / 0.000299792458f, 2.f) +
                            ::powf(p[2] / 0.000299792458f, 2.f)
                        ),
                        1.0f
                    )
        ));
    }
}

int main(int argc, char ** argv)
{
    boost::program_options::variables_map vm;
    parse_opts(argc, argv, vm);

    unsigned int width = vm["width"].as<unsigned int>();
    unsigned int height = vm["height"].as<unsigned int>();

    BOOST_LOG_TRIVIAL(info) << "Welcome to the covfie CUDA field renderer!";
    BOOST_LOG_TRIVIAL(info) << "Using magnetic field file \""
                            << vm["input"].as<std::string>() << "\"";
    BOOST_LOG_TRIVIAL(info) << "Starting read of input file...";

    std::ifstream ifs(vm["input"].as<std::string>(), std::ifstream::binary);

    if (!ifs.good()) {
        BOOST_LOG_TRIVIAL(fatal) << "Failed to open input file "
                                 << vm["input"].as<std::string>() << "!";
        std::exit(1);
    }

    cpu_field_t f(ifs);
    ifs.close();

    BOOST_LOG_TRIVIAL(info) << "Casting magnetic field into CUDA array...";

    cuda_field_t nf(covfie::make_parameter_pack(
        f.backend().get_configuration(), f.backend().get_backend().get_backend()
    ));

    BOOST_LOG_TRIVIAL(info) << "Allocating device memory for output image...";

    unsigned char * img_device;

    cudaErrorCheck(cudaMalloc(
        reinterpret_cast<void **>(&img_device),
        width * height * sizeof(unsigned char)
    ));

    BOOST_LOG_TRIVIAL(info) << "Rendering magnetic field strength to image...";

    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    dim3 dimBlock(32, 32);
    dim3 dimGrid(
        width / dimBlock.x + (width % dimBlock.x != 0 ? 1 : 0),
        height / dimBlock.y + (height % dimBlock.y != 0 ? 1 : 0)
    );

    render<decltype(nf)><<<dimGrid, dimBlock>>>(
        nf, img_device, width, height, vm["z"].as<float>()
    );

    cudaErrorCheck(cudaGetLastError());
    cudaErrorCheck(cudaDeviceSynchronize());

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();

    BOOST_LOG_TRIVIAL(info
    ) << "Rendering took "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << "us." << std::endl;

    BOOST_LOG_TRIVIAL(info) << "Allocating host memory for output image...";

    std::unique_ptr<unsigned char[]> img_host =
        std::make_unique<unsigned char[]>(width * height);

    BOOST_LOG_TRIVIAL(info) << "Copying image from device to host...";

    cudaErrorCheck(cudaMemcpy(
        img_host.get(),
        img_device,
        width * height * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));

    BOOST_LOG_TRIVIAL(info) << "Deallocating device memory...";

    cudaErrorCheck(cudaFree(img_device));

    BOOST_LOG_TRIVIAL(info) << "Saving image to file \""
                            << vm["output"].as<std::string>() << "\"...";

    render_bitmap(
        img_host.get(), width, height, vm["output"].as<std::string>()
    );

    BOOST_LOG_TRIVIAL(info) << "Procedure complete, goodbye!";
}
