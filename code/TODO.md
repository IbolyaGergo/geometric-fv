# Project TODOs

## Architecture & Extensibility
Switch from SecondOrderImplicit to HighResImplicit
- cfl > 0 -> df/du(`u_old[i]`) > 0

## Type Safety & Validation
- [ ] **Enhanced Type Hinting**: Use `numpy.typing.NDArray` for all array parameters to catch dimensional errors early.

## Logic & Cleanup
- [ ] **SolverState Management**: Add a `swap()` or `update_old()` method to `SolverState` to safely manage time-stepping.
