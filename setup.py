"""
HamiltonianPy setup file, used to install HamiltonianPy
"""


from setuptools import Extension, find_packages, setup


extension_path = "HamiltonianPy/extension/"
src_path = extension_path + "src/"
include_path = extension_path + "include/"

#compile_option = ["-fopenmp"]
#link_option = ["-lgomp"]
compile_option = None
link_option = None


c_files = [
    "matrix_repr_mod.c",
]

src = [src_path + c_file for c_file in c_files]
extension = Extension(
    name = "HamiltonianPy.extension.matrix_repr_c_mod",
    sources = src,
    depends = src,
    include_dirs = [include_path],
    extra_compile_args = compile_option,
    extra_link_args = link_option,
)


setup_params = dict(
    name="HamiltonianPy",
    version="1.1",
    description="Construct and solve a model Hamiltonian numerically!",
    author="wangshiphys",
    author_email="wangshiphys@gmail.com",
    long_description="Not available at this time!",
    keywords="Hamiltonian",
    url="https://github.com/wangshiphys/HamiltonianPy",
    python_requires='>=3.6',
    zip_safe=False,
    packages=find_packages(),
    ext_modules=[extension],
)


if __name__ == "__main__":
    setup(**setup_params)
