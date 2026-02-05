# Most important
Give answers that maximizes the development of the users critical thinking.

# Progress
The user is currently working on a module `src/example/sweep.py`.
I also tested the current code in `test/test_sweep.py`.
Before anything else, I want to code the solution of the implicit upwind method
for solving the linear advection equation with a positive speed a > 0
```tex
$ u_i^{n+1} = u_i^n - \nu ( u_i^{n+1} - u_{i-1}^{n+1} ) $
```
Start with u_{-1}^{n+1} = 0 inflow boundary condition, we solving for
`u_i^{n+1}` one by one.

# Requirements
The code must be maintainable and flexible.
We should utilize best practices in numerical computation using Python.
Look for inspiration in the most maintainable open source Python project solving
computational problems.
