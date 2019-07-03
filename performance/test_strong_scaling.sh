srun -n 1 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 2 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 4 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 8 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 16 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 32 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 64 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 128 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 256 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'
srun -n 512 python3 -c 'from test_perf_2d_parallel import test_perf_poisson_2d_1;test_perf_poisson_2d_1()'



