#include<cstdio>
#include<cmath>
#include<complex>
#include"sFFTenum.h"

namespace	sFFT {

	constexpr double twoPi = 2.0*M_PI;

	template<typename sFloat>
	class	sFFT {

		private:

		int		d;
		size_t		Nx, Ny, Nz;
		size_t		Gx, Gy, Gz;

		void		*data;
		void		*result;

		bool		useMPI;

		sFFTdir		sDir;
		sFFTtype	sType;

		void	iteraFFT (void *mData, void *rData, size_t M, size_t St) {

			if (M == 1) {
				static_cast<std::complex<sFloat> *>(rData)[0] = static_cast<std::complex<sFloat> *>(mData)[0];
				return;
			}

			size_t M2 = M >> 1;

			void *fft1 = static_cast<void*>(&static_cast<std::complex<sFloat> *>(mData)[0]);
			void *fft2 = static_cast<void*>(&static_cast<std::complex<sFloat> *>(mData)[St]);

			iteraFFT(fft1, static_cast<void *>(&static_cast<std::complex<sFloat> *>(rData)[0]),  M2, 2*St);
			iteraFFT(fft2, static_cast<void *>(&static_cast<std::complex<sFloat> *>(rData)[M2]), M2, 2*St);

/*	PSEUDOCODE

	Y[0,...,n−1]←recfft2(n,X,ι)
		IF n=1 THEN
			Y[0]←X[0]
		ELSE
			Y[0,...,n/2−1]←recfft2(n/2,X,2ι)
			Y[n/2,...,n−1]←recfft2(n/2,X+ι,2ι)

			FOR k1=0 TO (n/2)−1 DO
				t←Y[k1]
				Y[k1]    ← t + ωk1n Y[k1+n/2]
				Y[k1+n/2]← t − ωk1n Y[k1+n/2]
			ENDFOR
		ENDIF
	END
*/

			sFloat iM = ((sFloat) 1.0)/((sFloat) M);

			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < M2; i++) {

				std::complex<sFloat> Wk(cos(twoPi*((sFloat) i)*iM), sin(-twoPi*((sFloat) i)*iM));
				auto	tmp = static_cast<std::complex<sFloat> *>(rData)[i];
				auto	qmp = static_cast<std::complex<sFloat> *>(rData)[i+M2]*Wk;
				static_cast<std::complex<sFloat> *>(rData)[i]		= tmp + qmp;
				static_cast<std::complex<sFloat> *>(rData)[i+M2]	= tmp - qmp;
			}

			return;
		}

		public:

			sFFT  (int d, sFFTtype sType, sFFTdir sDir, bool useMPI, size_t Nx, size_t Ny, size_t Nz, void *data, void *result) : d(d), sType(sType), sDir(sDir), useMPI(useMPI), Nx(Nx), Ny(Ny), Nz(Nz), data(data), result(result) { }

		void	runFFT() {

			iteraFFT (data, result, Nx, 1);
			return;
		}
	};
}

