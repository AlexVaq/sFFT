#include<complex>
#include"sFFTenum.h"

void    runFFT  (int d, sFFT::sFFTtype sType, sFFT::sFFTdir sDir, bool useMPI, size_t Nx, size_t Ny, size_t Nz, void *data, void *result, sFFT::sFFTprec prec);
