/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

#include "cuda/utils/definitions.hpp"

/*
 * This structure contains all the information we need to use the magnetic field
 * on the GPU, at least in this simple context. It contains the texture in GPU
 * memory, the offsets in each direction (because texture indexing starts at
 * zero, but we can have negative coordinates), and the total size of the
 * magnetic field volume.
 */
struct MagneticField {
    /*
     * This is a CUDA object referring to a texture, the details are opaque and
     * we are not allowed to know what is inside.
     */
    cudaTextureObject_t texture;

    /*
     * The offsets for coordinates in the B-field volume.
     */
    float offset_x, offset_y, offset_z;

    /*
     * The total sizes of the B-field volume.
     */
    float size_x, size_y, size_z;
};

/*
 * This function takes the output of our kernel and writes it to a file, adding
 * the necessary headers to turn it into a bitmap image.
 */
void render_bitmap(char* img, unsigned int w, unsigned int h,
                   std::string fname) {
    /*
     * Start by opening our output file in binary format.
     */
    std::ofstream bmp(fname, std::ios::out | std::ios::binary);

    /*
     * The image size is the size of the image component (excluding the
     * headers), which is simply the product of the width and the height
     * because we are building a one-byte-per-pixel image.
     *
     * The width must be padded due to the way BMP images work.
     */
    unsigned int imgsize = (w + (4 - (w % 4)) % 4) * h;

    /*
     * The image data starts at byte 1078, after 54 bytes for the header and
     * 1024 bytes for the color palette.
     */
    unsigned int offset = 54 + 1024;

    /*
     * The total filesize is equal to the premable plus the image portion.
     */
    unsigned int filesize = offset + imgsize;

    /*
     * This is the definition of the BMP header... It's rather esoteric, but it
     * is necessary.
     */
    char header[54] = {
        /*
         * Bytes [0:1]: The BMP magic numbers.
         */
        'B',
        'M',
        /*
         * Bytes [2:5]: The total size of this file.
         */
        static_cast<char>(filesize),
        static_cast<char>(filesize >> 8),
        static_cast<char>(filesize >> 16),
        static_cast<char>(filesize >> 24),
        /*
         * Bytes [6:9]: Reserved bytes which nobody uses.
         */
        0,
        0,
        0,
        0,
        /*
         * Bytes [10:13]: The starting position of the image segment.
         */
        static_cast<char>(offset),
        static_cast<char>(offset >> 8),
        static_cast<char>(offset >> 16),
        static_cast<char>(offset >> 24),
        /*
         * Bytes [14:17]: This identifies the size of the DIB header, which in
         * this case identifies it as BITMAPINFOHEADER.
         */
        40,
        0,
        0,
        0,
        /*
         * Bytes [18:21]: The width of the image.
         */
        static_cast<char>(w),
        static_cast<char>(w >> 8),
        static_cast<char>(w >> 16),
        static_cast<char>(w >> 24),
        /*
         * Bytes [22:25]: The height of the image.
         */
        static_cast<char>(h),
        static_cast<char>(h >> 8),
        static_cast<char>(h >> 16),
        static_cast<char>(h >> 24),
        /*
         * Bytes [26:27]: The number of color planes, which is always 1.
         */
        1,
        0,
        /*
         * Bytes [28:29]: The number of bits per pixel, in this case 8 for one
         * byte.
         */
        8,
        0,
        /*
         * Bytes [30:53]: The rest of the header contains sensible defaults, so
         * we don't need to change them.
         */
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    };

    /*
     * Bitmap rows must be padded to multiples of 4, so we will keep these
     * padding zeros handy for when we need to do that.
     */
    char padding[3] = {0, 0, 0};

    /*
     * Start off by writing the 54 byte header to the file.
     */
    bmp.write(header, 54);

    /*
     * Here we compute the colour palette. Each byte identifies one of 256
     * colours, and this bit of code defines what those colors are. We use a
     * part of the HSV spectrum which looks nice. For an explanation of this
     * code look up the code for HSV to RGB conversion.
     */
    for (unsigned int i = 0; i < 256; ++i) {
        float s = 0.7;
        float v = 1.0;
        float hue = (i / 256.0) * 240.0;
        float c = v * s;
        float hp = hue / 60.0;

        float x = c * (1 - std::abs(std::fmod(hp, 2.0) - 1.0));

        float rp, gp, bp;

        if (hp < 1.0) {
            rp = c, gp = x, bp = 0;
        } else if (hp < 2.0) {
            rp = x, gp = c, bp = 0;
        } else if (hp < 3.0) {
            rp = 0, gp = c, bp = x;
        } else if (hp < 4.0) {
            rp = 0, gp = x, bp = c;
        } else if (hp < 5.0) {
            rp = x, gp = 0, bp = c;
        } else if (hp < 6.0) {
            rp = c, gp = 0, bp = x;
        }

        float q = v - c;

        /*
         * For each of the 256 colours, write the output RGB colour to the file
         * in four bytes.
         */
        char cmap[4] = {
            static_cast<char>((rp + q) * 255),
            static_cast<char>((gp + q) * 255),
            static_cast<char>((bp + q) * 255),
            0,
        };

        bmp.write(cmap, 4);
    }

    /*
     * Finally, write the image section. Start off at the bottom of the image
     * and work our way up.
     */
    for (std::size_t y = h - 1; y < h; --y) {
        for (std::size_t x = 0; x < w; ++x) {
            /*
             * Retrieve the magnitude from the image, and write it.
             */
            char r = img[h * x + y];
            bmp.write(&r, 1);
        }

        /*
         * If necessary, write the appropriate padding to the file.
         */
        bmp.write(padding, (4 - (w % 4)) % 4);
    }

    bmp.close();
}

/*
 * This is the meat of the pudding, but it is very simple. Given a magnetic
 * field and some coordinates, fetch the magnetic fiend at that point, possibly
 * doing trilinear interpolation.
 *
 * Returns a 4-vector, but only the first 3 values are used.
 */
__device__ float4 read_bfield(MagneticField bf, float x, float y, float z) {
    /*
     * We take the coordinates, offset them as appropriate, and add 0.5 which
     * is required by the interpolation hardware.
     */
    return tex3D<float4>(bf.texture, x + bf.offset_x + 0.5,
                         y + bf.offset_y + 0.5, z + bf.offset_z + 0.5);
}

/*
 * This kernel produces our output image.
 */
__global__ void bfield_img(MagneticField bf, float z, char* out, unsigned int w,
                           unsigned int h) {
    /*
     * This kernel should be called using a 2-dimensional block and grid, so we
     * find this thread's x and y coordinates.
     */
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    /*
     * Bounds checking.
     */
    if (x >= w || y >= h) {
        return;
    }

    /*
     * Compute our normalized position in the image.
     */
    float xf = x / static_cast<float>(w);
    float yf = y / static_cast<float>(h);

    /*
     * Compute the corresponding position in the magnetic field.
     */
    float xc = -bf.offset_x + xf * bf.size_x;
    float yc = -bf.offset_y + yf * bf.size_y;

    /*
     * Call our retrieval function to get the value of the magnetic field at
     * this point.
     */
    float4 r = read_bfield(bf, xc, yc, z);

    /*
     * Compute the magnitude of our magnetic field at this position, so we can
     * actually plot it.
     */
    float m = std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z);

    /*
     * Write the output to the array, making sure to clamp the value
     * appropriately.
     */
    out[h * x + y] = 255 * min(m, 1.0f);
}

int main(int argc, char* argv[]) {
    /*
     * Some argument parsing code, should speak for itself.
     */
    if (argc < 2) {
        std::cout << "Not enough arguments, minimum requirement: " << std::endl;
        std::cout << "./bfield_cuda <bfield.txt> [z=0.0] [w=1024] [h=1024]"
                  << std::endl;
        return -1;
    }

    std::string bfield_map_file = std::string(argv[1]);

    float z_value = 0.0;
    unsigned int w = 1024;
    unsigned int h = 1024;

    if (argc >= 3) {
        z_value = std::atof(argv[2]);
    }

    if (argc >= 4) {
        w = std::atoi(argv[3]);
    }

    if (argc >= 5) {
        h = std::atoi(argv[4]);
    }

    /*
     * We will start by computing the limits of our magnetic field, which is to
     * say how big it is. This step is not necessary in a more optimized
     * magnetic field format, but we will need to do it like this for now.
     */
    std::ifstream f;

    std::cout << "Computing magnetic field limits..." << std::endl;

    long minx = std::numeric_limits<long>::max();
    long maxx = std::numeric_limits<long>::lowest();
    long miny = std::numeric_limits<long>::max();
    long maxy = std::numeric_limits<long>::lowest();
    long minz = std::numeric_limits<long>::max();
    long maxz = std::numeric_limits<long>::lowest();

    {
        f.open(bfield_map_file);

        float xp, yp, zp;
        float Bx, By, Bz;

        /*
         * Read every line, and update our current minima and maxima
         * appropriately.
         */
        while (f >> xp >> yp >> zp >> Bx >> By >> Bz) {
            long x = std::lround(xp / 100.0f);
            long y = std::lround(yp / 100.0f);
            long z = std::lround(zp / 100.0f);

            minx = std::min(minx, x);
            maxx = std::max(maxx, x);

            miny = std::min(miny, y);
            maxy = std::max(maxy, y);

            minz = std::min(minz, z);
            maxz = std::max(maxz, z);
        }

        f.close();
    }

    /*
     * Now that we have the limits of our field, compute the size in each
     * dimension.
     */
    int sx = (maxx - minx) + 1;
    int sy = (maxy - miny) + 1;
    int sz = (maxz - minz) + 1;

    /*
     * Now, we compute the stride of our dimensions.
     */
    int tx = 4;
    int ty = tx * sx;
    int tz = ty * sy;

    /*
     * Finally, we need the offsets of our magnetic field because the indexing
     * must start at 0.
     */
    int ox = -minx;
    int oy = -miny;
    int oz = -minz;

    /*
     * Next, we allocate an array on the host, which we will use to construct
     * our magnetic field.
     */
    float* ptr = static_cast<float*>(malloc(sx * sy * sz * 4 * sizeof(float)));

    /*
     * The next step is to read the file again, but this time we will insert
     * each value at the appropriate position in the array.
     */
    std::cout << "Arranging magnetic field in 3D array..." << std::endl;

    {
        f.open(bfield_map_file);

        float xp, yp, zp;
        float Bx, By, Bz;

        while (f >> xp >> yp >> zp >> Bx >> By >> Bz) {
            long x = std::lround(xp / 100.0f);
            long y = std::lround(yp / 100.0f);
            long z = std::lround(zp / 100.0f);

            /*
             * Compute the correct 4-vector to write to.
             */
            float* dst = &ptr[(x + ox) * tx + (y + oy) * ty + (z + oz) * tz];

            /*
             * Write the direction of the magnetic field to the array, using
             * the w-coordinate as a dummy value.
             */
            dst[0] = Bx;
            dst[1] = By;
            dst[2] = Bz;
            dst[3] = 0;
        }

        f.close();
    }

    /*
     * Finally, we can start to construct the magnetic field on the GPU.
     */
    std::cout << "Constructing magnetic field on the GPU..." << std::endl;

    /*
     * This is a struct of our own design, which will hold some administrative
     * data.
     */
    MagneticField bfield;

    bfield.offset_x = -minx;
    bfield.offset_y = -miny;
    bfield.offset_z = -minz;

    bfield.size_x = sx;
    bfield.size_y = sy;
    bfield.size_z = sz;

    {
        /*
         * This is a CUDA builtin which describes what the channel (the value
         * for each pixel) should look like. I think this is only defined for
         * float, float2, and float4, so we will need to use float4 here.
         */
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        /*
         * CUDA extents are basically just multi-dimensional sizes, but we need
         * to define how big our texture is.
         */
        cudaExtent extent = make_cudaExtent(sx, sy, sz);

        /*
         * This represents an opaque array on the device, with an unknown
         * storage configuration. The idea is that CUDA will use one that is
         * efficient for the spatial locality that is usually found in
         * texture-related problems.
         */
        cudaArray_t content;

        /*
         * Make sure we allocate the memory for this array, or it won't work.
         */
        CUDA_ERROR_CHECK(cudaMalloc3DArray(&content, &channelDesc, extent));

        /*
         * Next, we must define the copy settings for moving the B-field data
         * from the host to the device. Because this has so many options, we
         * need a struct for this.
         */
        cudaMemcpy3DParms copyParams = {0};

        /*
         * The source pointer is a pitched host pointer.
         */
        copyParams.srcPtr =
            make_cudaPitchedPtr(ptr, sx * sizeof(float4), sx, sy);

        /*
         * The destination is our device array.
         */
        copyParams.dstArray = content;

        /*
         * The size to be copied is equal to the size of our field.
         */
        copyParams.extent = extent;

        /*
         * This is the same enum that we know and love from cudaMemcpy, but as
         * a struct member this time.
         */
        copyParams.kind = cudaMemcpyHostToDevice;

        /*
         * Finally, we can perform the copy.
         */
        CUDA_ERROR_CHECK(cudaMemcpy3D(&copyParams));

        /*
         * Next up, we need to describe where CUDA can find our texture data.
         * This needs to be manually zeroed because apparently CUDA is from the
         * year 1971...
         */
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));

        /*
         * Inform CUDA that we are reading from an array (not a pointer!) and
         * tell it where to find this array.
         */
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = content;

        /*
         * Finally, we need to describe how our texture is supposed to work.
         * Again, we zero this struct.
         */
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));

        /*
         * First of all, we need to set the out-of-bounds mechanism. Most
         * sensible for this application seems to be to clamp in all dimensions.
         * An r-phi implementation may want to use a wrapping approach instead!
         */
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;

        /*
         * Importantly, this flag enables the trilinear interpolation of our
         * texture.
         */
        texDesc.filterMode = cudaFilterModeLinear;

        /*
         * This flag disables the value normalization which we do not want.
         */
        texDesc.readMode = cudaReadModeElementType;

        /*
         * Given all of this configuration, FINALLY construct our texture,
         * directly into our magnetic field object.
         */
        CUDA_ERROR_CHECK(cudaCreateTextureObject(&bfield.texture, &resDesc,
                                                 &texDesc, nullptr));
    }

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * Now we can finally start doing fun stuff and computing things on the
       GPU.
     */

    std::cout << "Rendering image..." << std::endl;

    /*
     * This array will hold our output image, with each pixel representing the
     * magnitude of the B-field at that position, in the range [0, 256).
     */
    char* img;

    CUDA_ERROR_CHECK(cudaMallocManaged(&img, w * h));

    /*
     * This is a 2D kernel, so we will need to define the block and the grid
     * sizes.
     */
    dim3 block_size(16, 16);
    dim3 grid_size(w / 16 + (w % 16 > 0 ? 1 : 0),
                   h / 16 + (h % 16 > 0 ? 1 : 0));

    /*
     * Launch the kernel.
     */
    bfield_img<<<grid_size, block_size>>>(bfield, z_value, img, w, h);

    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /*
     * And now all we need to do is write the image to the file.
     */
    std::string out_file = "output.bmp";

    std::cout << "Saving bitmap to " << out_file << "..." << std::endl;

    /*
     * All the bitmap writing code is encapsulated, thank god.
     */
    render_bitmap(img, w, h, out_file);

    std::cout << "Job complete. Goodbye!" << std::endl;

    return 0;
}
