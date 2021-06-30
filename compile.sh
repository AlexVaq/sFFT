#!/bin/bash

g++-mp-10 -m64 --std=c++17 -fopenmp -O3 -Iinclude -I/usr/local/include -L/usr/local/lib -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f lib/sFFT.cpp test/FFTtest.cpp -o test/FFTtest
