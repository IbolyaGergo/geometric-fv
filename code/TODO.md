# Project TODOs

## Architecture & Extensibility
Switch from SecondOrderImplicit to HighResImplicit
- cfl > 0 -> df/du(`u_old[i]`) > 0

## Type Safety & Validation
- [ ] **Enhanced Type Hinting**: Use `numpy.typing.NDArray` for all array parameters to catch dimensional errors early.
- SolverState.niter should be moved elsewhere. E.g., creating a stats dataclas.
  There will be niter together with errors and other statistics.

## Logic & Cleanup
- [ ] **SolverState Management**: Add a `swap()` or `update_old()` method to `SolverState` to safely manage time-stepping.
