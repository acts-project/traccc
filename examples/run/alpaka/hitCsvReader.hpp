#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <istream>
 
 
std::vector<std::string> csv_read_row(const std::istream &in, char delimiter);
std::vector<std::string> csv_read_row(const std::ifstream &in, char delimiter);

namespace traccc
{
    class hitCsvReader
    {
    public:
        hitCsvReader(std::string inputDir, uint event);

        struct cells {
            u_int64_t geoID[500000];
            u_int16_t channel0[500000];
            u_int16_t channel1[500000];

            // std::vector<u_int64_t> geoID;
            // std::vector<u_int16_t> channel0;
            // std::vector<u_int16_t> channel1;
        };

        cells data;
    };

    hitCsvReader::hitCsvReader(std::string inputDir, uint event)
    {   
        using std::cout;
        using std::endl;

        std::ifstream in(inputDir + "event00000000" + std::to_string(event) + "-cells.csv");
        char buffer[1024];
        u_int lineNum = 0;
        
        in.getline(buffer, 1024);

        if (!in.fail()) {
        while(!in.eof())
        {   
            unsigned long int a;
            int dummy;
            unsigned short c0, c1;
            int d;
            float dummy2;
            char dl;

            in>>a>>dl>>dummy>>dl>>c0>>dl>>c1>>dl>>d>>dl>>dummy2;

            data.geoID[lineNum] = a;
            data.channel0[lineNum] = c0;
            data.channel1[lineNum] = c1;

            // data.geoID.push_back(a);
            // data.channel0.push_back(c0);
            // data.channel1.push_back(c1);
        
            ++lineNum;
        }
        in.close();

        }
    }
    
   
} // namespace traccc