"""
HamiltonianPy setup file, used to install HamiltonianPy.
"""

from setuptools import Extension, find_packages, setup

cext_path = 'HamiltonianPy/matrepr/extension/cext/'
src_path = cext_path + 'src/'
include_path = cext_path + 'include/'
parent_pkg = 'HamiltonianPy.matrepr.'

compile_option = ['-fopenmp']
link_option = ['-lgomp']
#compile_option = None
#link_option = None

src1=[src_path + 'bisearch.c', src_path + 'numof1.c', src_path + 'repr.c', 
      src_path + 'convert.c', src_path + 'reprmod.c']
ext1 = Extension(name=parent_pkg + 'matreprcext',
                sources=src1,
                depends=src1,
                include_dirs=[include_path],
                extra_compile_args = compile_option,
                extra_link_args = link_option,
                )

src2=[src_path + 'bisearch.c', src_path + 'convert.c', 
      src_path + 'bisearchmod.c']
ext2 = Extension(name=parent_pkg + 'bisearch',
                sources=src2,
                depends=src2,
                include_dirs=[include_path],
                extra_compile_args = compile_option,
                extra_link_args = link_option,
                )

src3=[src_path + 'numof1.c', src_path + 'convert.c', src_path + 'numof1mod.c']
ext3 = Extension(name=parent_pkg + 'numof1',
                sources=src3,
                depends=src3,
                include_dirs=[include_path],
                extra_compile_args = compile_option,
                extra_link_args = link_option,
                )

src4=[src_path + 'convert.c', src_path + 'bitselectmod.c']
ext4 = Extension(name=parent_pkg + 'bitselect',
                sources=src4,
                depends=src4,
                include_dirs=[include_path],
                extra_compile_args = compile_option,
                extra_link_args = link_option,
                )


setup_params = dict(
    name="HamiltonianPy",
    version="1.0.dev0",
    description="Construct and solve a model Hamiltonian numerically!",
    author="wangshiphys",
    author_email="wangshiphys@gmail.com",
    long_description="Not availiable at this time!",
    keywords="Hamiltonian",
    url="Not availiable at this time!",
    python_requires='>=3.0',
    zip_safe=False,
    packages=find_packages(),
    ext_modules=[ext1, ext2, ext3, ext4],
)

if __name__ == "__main__":
    setup(**setup_params)
