from spack.package import *


class Covfie(CMakePackage):
    homepage = "https://github.com/acts-project/covfie"
    git = "https://github.com/acts-project/covfie.git"
    list_url = "https://github.com/acts-project/covfie/tags"

    version("0.5.0", commit="59bd2e0a4861db6e5898606e0167da895505e46b")

    variant("concepts", default=False, description="Enforce C++20 concepts")
    variant("cuda", default=False, description="Enables the CUDA platform")

    depends_on("cmake@3.18:", type="build")

    def cmake_args(self):
        spec = self.spec

        args = [
            self.define("COVFIE_PLATFORM_CPU", True),
            self.define_from_variant("COVFIE_PLATFORM_CUDA", "cuda"),
            self.define_from_variant("COVFIE_REQUIRE_CXX20", "concepts"),
            self.define("COVFIE_QUIET", True),
        ]

        return args
