"""
HamiltonianPy setup file, used to install HamiltonianPy.
"""

from setuptools import Extension, find_packages, setup

cext_path = "HamiltonianPy/extpkg/cext/"
src_path = cext_path + "src/"
include_path = cext_path + "include/"
parent_pkg = "HamiltonianPy.extpkg."

#compile_option = ["-fopenmp"]
#link_option = ["-lgomp"]
compile_option = None
link_option = None

cfiles=["matrixreprmod.c"]
src=[src_path + cfile for cfile in cfiles]
ext = Extension(name=parent_pkg + 'matrixrepr',
                sources=src,
                depends=src,
                include_dirs=[include_path],
                extra_compile_args = compile_option,
                extra_link_args = link_option,
                )

setup_params = dict(
    name="HamiltonianPy",
    version="1.0.dev1",
    description="Construct and solve a model Hamiltonian numerically!",
    author="wangshiphys",
    author_email="wangshiphys@gmail.com",
    long_description="Not availiable at this time!",
    keywords="Hamiltonian",
    url="Not availiable at this time!",
    python_requires='>=3.0',
    zip_safe=False,
    packages=find_packages(),
    ext_modules=[ext],
)

if __name__ == "__main__":
    setup(**setup_params)
