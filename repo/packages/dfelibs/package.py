from spack.package import *


class Dfelibs(CMakePackage):
    homepage = "https://github.com/acts-project/dfelibs"
    git = "https://github.com/acts-project/dfelibs.git"
    list_url = "https://github.com/acts-project/dfelibs/tags"

    version("2021.10.29", commit="2160cefb88cbaea9bb023bef6c1ba6549c139cf6")

    depends_on("cmake@3.8:", type="build")
