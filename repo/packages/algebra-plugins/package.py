from spack.package import *


class AlgebraPlugins(CMakePackage):
    homepage = "https://github.com/acts-project/algebra-plugins"
    git = "https://github.com/acts-project/algebra-plugins.git"
    list_url = "https://github.com/acts-project/algebra-plugins/tags"

    # Older versions are not supported due to build system bugs.
    version("0.18.0", commit="d5def778f446451391641d3316d0c47c85e1d5de")

    variant(
        "cxxstd",
        default="17",
        values=("17", "20", "23"),
        multi=False,
        description="C++ standard used",
    )
    variant("eigen", default=False, description="Enables the Eigen plugin")
    variant("smatrix", default=False, description="Enables the SMatrix plugin")
    variant("vecmem", default=False, description="Enables the vecmem plugin")
    variant("vc", default=False, description="Enables the Vc plugin")

    depends_on("cmake@3.14:", type="build")
    depends_on("vecmem@0.22.0: +cuda", when="+vecmem")
    depends_on("eigen@3.4.0:", when="+eigen")
    depends_on("vc@1.4.0:", when="+vc")
    depends_on("root@6.18.0:", when="+smatrix")

    def cmake_args(self):
        spec = self.spec

        args = [
            self.define_from_variant("CMAKE_CXX_STANDARD", "cxxstd"),
            self.define("ALGEBRA_PLUGINS_USE_SYSTEM_LIBS", True),
            self.define_from_variant("ALGEBRA_PLUGINS_INCLUDE_EIGEN", "eigen"),
            self.define_from_variant("ALGEBRA_PLUGINS_INCLUDE_SMATRIX", "smatrix"),
            self.define_from_variant("ALGEBRA_PLUGINS_INCLUDE_VC", "vc"),
            self.define_from_variant("ALGEBRA_PLUGINS_INCLUDE_VECMEM", "vecmem"),
        ]

        return args
