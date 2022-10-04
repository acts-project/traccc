#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>

int main( int argc, char** argv ) {
  Kokkos::initialize( argc, arvc );
  {
    testing::InitGoogleTest(&argc, argv);
    int r = RUN_ALL_TESTS();
  }
  Kokkos::finalize();
  return 0;
}
