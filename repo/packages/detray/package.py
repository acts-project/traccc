from spack.package import *


class Detray(CMakePackage):
    homepage = "https://github.com/acts-project/detray"
    git = "https://github.com/acts-project/detray.git"
    list_url = "https://github.com/acts-project/detray/tags"

    # Older versions are not supported due to build system bugs.
    version("0.28.0", commit="a4ae8c45edf8b1adadef37bd33046215d611f1d4")

    variant("actsvg", default=False, description="Enables the actsvg plugin")
    variant("csv", default=True, description="Enable the CSV IO plugin")
    variant(
        "cxxstd",
        default="17",
        values=("17", "20", "23"),
        multi=False,
        description="C++ standard used",
    )
    variant("json", default=True, description="Enable the JSON IO plugin")
    variant(
        "scalar",
        default="float",
        values=("float", "double"),
        multi=False,
        description="Scalar type to use by default",
    )
    variant("eigen", default=True, description="Enable the Eigen math plugin")
    variant("smatrix", default=False, description="Enable the SMatrix math plugin")
    variant("vc", default=True, description="Enable the Vc math plugin")

    depends_on("cmake@3.11:", type="build")
    depends_on("vecmem@0.22.0:")
    depends_on("covfie@0.5.0:")
    depends_on("nlohmann-json@3.11.0:", when="+json")
    depends_on("dfelibs@2021.10.29")
    depends_on("algebra-plugins@0.18.0: +vecmem")
    depends_on("algebra-plugins +vc", when="+vc")
    depends_on("algebra-plugins +eigen", when="+eigen")
    depends_on("algebra-plugins +smatrix", when="+smatrix")
    depends_on("actsvg +meta", when="+actsvg")

    def cmake_args(self):
        args = [
            self.define("DETRAY_USE_SYSTEM_LIBS", True),
            self.define_from_variant("CMAKE_CXX_STANDARD", "cxxstd"),
            self.define_from_variant("CMAKE_CUDA_STANDARD", "cxxstd"),
            self.define_from_variant("CMAKE_SYCL_STANDARD", "cxxstd"),
            self.define_from_variant("DETRAY_SETUP_ACTSVG", "actsvg"),
            self.define_from_variant("DETRAY_CUSTOM_SCALARTYPE", "scalar"),
            self.define_from_variant("DETRAY_EIGEN_PLUGIN", "eigen"),
            self.define_from_variant("DETRAY_SMATRIX_PLUGIN", "smatrix"),
            self.define_from_variant("DETRAY_IO_CSV", "csv"),
            self.define_from_variant("DETRAY_IO_JSON", "json"),
        ]

        return args
