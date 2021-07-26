#include"sFFT.h"

void	runFFT		(int d, sFFT::sFFTtype sType, sFFT::sFFTdir sDir, bool useMPI, size_t Nx, size_t Ny, size_t Nz, void *data, void *result, sFFT::sFFTprec prec) {

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

void	runButterFFT	(int d, sFFT::sFFTtype sType, sFFT::sFFTdir sDir, bool useMPI, size_t Nx, size_t Ny, size_t Nz, void *data, void *result, sFFT::sFFTprec prec, int radix) {

	switch (prec) {
		case sFFT::sFFT_Double: {
			switch (radix) {
				case 2: {
					sFFT::sFFT<double,1> myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
				case 4: {
					sFFT::sFFT<double,2> myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
				case 8: {
					sFFT::sFFT<double,3> myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
				case 16: {
					sFFT::sFFT<double,4> myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
			}
			break;
		}

		case sFFT::sFFT_Single: {
			switch (radix) {
				case 2: {
					sFFT::sFFT<float,1>  myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
				case 4: {
					sFFT::sFFT<float,2>  myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
				case 8: {
					sFFT::sFFT<float,3>  myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
				case 16: {
					sFFT::sFFT<float,4>  myFFT(d, sType, sDir, useMPI, Nx, Ny, Nz, data, result);
					myFFT.runButterFFT();
					break;
				}
			}
		}

		default:
			break;
	}
}
