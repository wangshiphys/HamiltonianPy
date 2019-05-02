# Ideas about this project

---

## Realized

- [x] Add `NumberOperator` class to `termofH` module
- [x] Add some factory functions to `termofH` module for generating some commonly used Hamiltonian term
- [x] Add more detailed data for some commonly used lattice, especially the high-symmetry points in the first-Brillouin-Zone
- [x] Add `return_indices` parameter to the `KPath` function defined in `lattice` module
- [x] Test the efficience of `scipy.sparse.linalg.inv` function
- [x] Test the efficience of solving linear problems of large sparse matrix

## TODO

- [ ] Develop a `TBA` module for processing tight-binding related problems
- [ ] Add some demo scripts to components of this project
- [ ] Optimize the implementation of `ParticleConservedExactSolver` and `ParticleNotConversedExactSolver` function, change from loop implementation to matrix operation implementation for calculating the cluster Green-Function
- [ ] For some commonly used Hamiltonian term, provide more specific as well as efficient implementation for calculating the matrix representation
- [ ] Rename the `termofH` module to `QuantumOperator`
