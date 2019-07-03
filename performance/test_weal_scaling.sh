srun -n 1 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_2;test_perf_poisson_2d_2()'
srun -n 4 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_2;test_perf_poisson_2d_2()'
srun -n 16 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_2;test_perf_poisson_2d_2()'
srun -N 64 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_2;test_perf_poisson_2d_2()'
srun -N 256 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_2;test_perf_poisson_2d_2()'



