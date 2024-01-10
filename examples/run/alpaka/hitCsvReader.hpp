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
        hitCsvReader(std::string fileName);

        struct cells {
            u_long geoID[55000];
            u_short channel0[55000];
            u_short channel1[55000];
        };

        cells data;
    };

    hitCsvReader::hitCsvReader(std::string fileName)
    {   
        using std::cout;
        using std::endl;

        std::ifstream in(fileName);
        char buffer[1024];
        u_int numLines = 0;
        
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

            // std::cout<<a<<","<<c0<<","<<c1<<std::endl;
            data.geoID[numLines] = a;
            data.channel0[numLines] = c0;
            data.channel1[numLines] = c1;
        
            ++numLines;
        }
        in.close();

        }
    }
    
   
} // namespace traccc