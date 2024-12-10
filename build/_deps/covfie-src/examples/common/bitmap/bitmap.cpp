/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cmath>
#include <fstream>
#include <string>

#include "bitmap.hpp"

/*
 * This function takes the output of our kernel and writes it to a file, adding
 * the necessary headers to turn it into a bitmap image.
 */
void render_bitmap(
    unsigned char * img, unsigned int w, unsigned int h, std::string fname
)
{
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
    unsigned char header[54] = {
        /*
         * Bytes [0:1]: The BMP magic numbers.
         */
        'B',
        'M',
        /*
         * Bytes [2:5]: The total size of this file.
         */
        static_cast<unsigned char>(filesize),
        static_cast<unsigned char>(filesize >> 8),
        static_cast<unsigned char>(filesize >> 16),
        static_cast<unsigned char>(filesize >> 24),
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
        static_cast<unsigned char>(offset),
        static_cast<unsigned char>(offset >> 8),
        static_cast<unsigned char>(offset >> 16),
        static_cast<unsigned char>(offset >> 24),
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
        static_cast<unsigned char>(w),
        static_cast<unsigned char>(w >> 8),
        static_cast<unsigned char>(w >> 16),
        static_cast<unsigned char>(w >> 24),
        /*
         * Bytes [22:25]: The height of the image.
         */
        static_cast<unsigned char>(h),
        static_cast<unsigned char>(h >> 8),
        static_cast<unsigned char>(h >> 16),
        static_cast<unsigned char>(h >> 24),
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
    unsigned char padding[3] = {0, 0, 0};

    /*
     * Start off by writing the 54 byte header to the file.
     */
    bmp.write(reinterpret_cast<char *>(header), 54);

    /*
     * Here we compute the colour palette. Each byte identifies one of 256
     * colours, and this bit of code defines what those colors are. We use a
     * part of the HSV spectrum which looks nice. For an explanation of this
     * code look up the code for HSV to RGB conversion.
     */
    for (unsigned int i = 0; i < 256; ++i) {
        float s = 0.7f;
        float v = 1.0f;
        float hue = (static_cast<float>(i) / 256.0f) * 240.0f;
        float c = v * s;
        float hp = hue / 60.0f;

        float x = c * (1.0f - std::abs(std::fmod(hp, 2.0f) - 1.0f));

        float rp = 0.f, gp = 0.f, bp = 0.f;

        if (hp < 1.0f) {
            rp = c, gp = x, bp = 0;
        } else if (hp < 2.0f) {
            rp = x, gp = c, bp = 0;
        } else if (hp < 3.0f) {
            rp = 0, gp = c, bp = x;
        } else if (hp < 4.0f) {
            rp = 0, gp = x, bp = c;
        } else if (hp < 5.0f) {
            rp = x, gp = 0, bp = c;
        } else if (hp < 6.0f) {
            rp = c, gp = 0, bp = x;
        }

        float q = v - c;

        /*
         * For each of the 256 colours, write the output RGB colour to the file
         * in four bytes.
         */
        unsigned char cmap[4] = {
            static_cast<unsigned char>((rp + q) * 255),
            static_cast<unsigned char>((gp + q) * 255),
            static_cast<unsigned char>((bp + q) * 255),
            0,
        };

        bmp.write(reinterpret_cast<char *>(cmap), 4);
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
            unsigned char r = img[h * x + y];
            bmp.write(reinterpret_cast<char *>(&r), 1);
        }

        /*
         * If necessary, write the appropriate padding to the file.
         */
        bmp.write(reinterpret_cast<char *>(padding), (4 - (w % 4)) % 4);
    }

    bmp.close();
}
