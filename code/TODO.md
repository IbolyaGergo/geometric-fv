# Project TODOs

# Clean up HighResImplicit
* remove the repetitions

## Type Safety & Validation
- [ ] **Enhanced Type Hinting**: Use `numpy.typing.NDArray` for all array parameters to catch dimensional errors early.
- SolverState.niter should be moved elsewhere. E.g., creating a stats dataclass.
  There will be niter together with errors and other statistics.
