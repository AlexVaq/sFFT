#include"sFFT.h"

void	runFFT	(int d, sFFT::sFFTtype sType, sFFT::sFFTdir sDir, bool useMPI, size_t Nx, size_t Ny, size_t Nz, void *data, void *result, sFFT::sFFTprec prec) {

	switch (prec) {
		case sFFT::sFFT_Double: {
			sFFT::sFFT<double> myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
			myFFT.runFFT();
			break;
		}

		case sFFT::sFFT_Single: {
			sFFT::sFFT<float>  myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
			myFFT.runFFT();
			break;
		}

		default:
			break;
	}
}
