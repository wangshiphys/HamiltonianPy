"""
HamiltonianPy setup file, used to install HamiltonianPy.
"""


from setuptools import find_packages, setup


setup_params = dict(
    name="HamiltonianPy",
    version="2.0.0",
    description="Construct and solve a model Hamiltonian numerically!",
    author="wangshiphys",
    author_email="wangshiphys@gmail.com",
    long_description="Not available at this time!",
    keywords="Hamiltonian",
    url="https://github.com/wangshiphys/HamiltonianPy",
    python_requires='>=3.6',
    zip_safe=False,
    packages=find_packages(),
)


if __name__ == "__main__":
    setup(**setup_params)
