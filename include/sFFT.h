#include<cstdio>
#include<cmath>
#include<complex>
#include<array>
#include"sFFTenum.h"

#include<cstring>

namespace	sFFT {

	typedef	unsigned int uint;
	constexpr double twoPi = 2.0*M_PI;

	constexpr std::array<unsigned char, 256> b = {   0, 128, 64, 192, 32, 160,  96, 224, 16, 144, 80, 208, 48, 176, 112, 240,  8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
							 4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
							 2, 130, 66, 194, 34, 162,  98, 226, 18, 146, 82, 210, 50, 178, 114, 242, 10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
							 6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246, 14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
							 1, 129, 65, 193, 33, 161,  97, 225, 17, 145, 81, 209, 49, 177, 113, 241,  9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
							 5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245, 13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
							 3, 131, 67, 195, 35, 163,  99, 227, 19, 147, 83, 211, 51, 179, 115, 243, 11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
							 7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247, 15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255 };

	template<typename sFloat, uint radix=1>
	class	sFFT {

		private:

		const uint	rSize;

		int		d;
		uint		Nx, Ny, Nz;
		uint		Gx, Gy, Gz;
		uint		lx, ly, lz;

		void		*data;
		void		*result;

		bool		useMPI;

		sFFTdir		sDir;
		sFFTtype	sType;

		/*	PSEUDOCODE ITERATIVE FFT

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
		void	iteraFFT (void *mData, void *rData, uint M, uint St) {

			if (M == 1) {
				static_cast<std::complex<sFloat> *>(rData)[0] = static_cast<std::complex<sFloat> *>(mData)[0];
				return;
			}

			uint M2 = M >> 1;

			void *fft1 = static_cast<void*>(&static_cast<std::complex<sFloat> *>(mData)[0]);
			void *fft2 = static_cast<void*>(&static_cast<std::complex<sFloat> *>(mData)[St]);

			iteraFFT(fft1, static_cast<void *>(&static_cast<std::complex<sFloat> *>(rData)[0]),  M2, 2*St);
			iteraFFT(fft2, static_cast<void *>(&static_cast<std::complex<sFloat> *>(rData)[M2]), M2, 2*St);


			sFloat iM = ((sFloat) 1.0)/((sFloat) M);

			#pragma omp parallel for schedule(static)
			for (uint i = 0; i < M2; i++) {

				std::complex<sFloat> Wk(cos(twoPi*((sFloat) i)*iM), sin(-twoPi*((sFloat) i)*iM));
				auto	tmp = static_cast<std::complex<sFloat> *>(rData)[i];
				auto	qmp = static_cast<std::complex<sFloat> *>(rData)[i+M2]*Wk;
				static_cast<std::complex<sFloat> *>(rData)[i]		= tmp + qmp;
				static_cast<std::complex<sFloat> *>(rData)[i+M2]	= tmp - qmp;
			}

			return;
		}

		static unsigned int rw(uint k) {

			unsigned char	b0 = b[k >> 0*8 & 0xff];
 			unsigned char	b1 = b[k >> 1*8 & 0xff];
 			unsigned char	b2 = b[k >> 2*8 & 0xff];
 			unsigned char	b3 = b[k >> 3*8 & 0xff];

			return	b0 << 3*8 | b1 << 2*8 | b2 << 1*8 | b3 << 0*8;
		}

		static float r(uint k) {
			return 1./4294967296. * rw(k);
		}

		void	butterFFTLast (void *mData, void *rData, uint u0) { // Only for radix 4

			sFloat	a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i,
					  b1r, b1i, b2r, b2i, b3r, b3i,
				c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i,
				d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i;

			for (int k0 = 0; k0 < u0; ++k0) {

				CommonWeight weight = weights[k0];

				a0r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 0].real();
				a0i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 0].imag();
				a1r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 1].real();
				a1i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 1].imag();
				a2r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 2].real();
				a2i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 2].imag();
				a3r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 3].real();
				a3i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 3].imag();

				b1r = - a1i * weight.w1i + a1r;
				b1i = + a1r * weight.w1i + a1i;
				b2r = - a2i * weight.w2i + a2r;
				b2i = + a2r * weight.w2i + a2i;
				b3r = - a3i * weight.w3i + a3r;
				b3i = + a3r * weight.w3i + a3i;
				c0r = + b2r * weight.w2r + a0r;
				c0i = + b2i * weight.w2r + a0i;
				c2r = - b2r * weight.w2r + a0r;
				c2i = - b2i * weight.w2r + a0i;
				c1r = + b3r * weight.w3r + b1r;
				c1i = + b3i * weight.w3r + b1i;
				c3r = - b3r * weight.w3r + b1r;
				c3i = - b3i * weight.w3r + b1i;
				d0r = + c1r * weight.w1r + c0r;
				d0i = + c1i * weight.w1r + c0i;
				d1r = - c1r * weight.w1r + c0r;
				d1i = - c1i * weight.w1r + c0i;
				d2r = - c3i * weight.w1r + c2r;
				d2i = + c3r * weight.w1r + c2i;
				d3r = + c3i * weight.w1r + c2r;
				d3i = - c3r * weight.w1r + c2i;

				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 0].real() = d0r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 0].imag() = d0i;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 1].real() = d1r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 1].imag() = d1i;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 2].real() = d2r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 2].imag() = d2i;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 3].real() = d3r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 3].imag() = d3i;
			}

			return;
		}

		void	butterFFT (void *mData, void *rData, uint k0, uint c0) {

			const uint c1	= c0 >> radix;

			for (uint k2 = 0; k2 < c1; k2++)
				for (uint k1 = 0; k1 < rSize; k1++) {
					std::complex<sFloat> sm(0.,0.);

					for (int j1 = 0; j1 < rSize; j1++) {
						auto arg0 = twoPi*j1*r(k0*rSize);
						auto arg1 = twoPi*j1*r(k1);
						std::complex<sFloat> w0(std::cos(arg0), std::sin(arg0));
						std::complex<sFloat> w1(std::cos(arg1), std::sin(arg1));

						sm +=  w0 * w1 * static_cast<std::complex<sFloat> *>(mData)[c0*k0 + c1*j1 + k2];
auto caca = static_cast<std::complex<sFloat> *>(mData)[c0*k0 + c1*j1 + k2];
printf("%+.3lf %+.3lf | %+.3lf %+.3lf | %+.3lf %+.3lf\n", w0.real(), w0.imag(), w1.real(), w1.imag(), caca.real(), caca.imag()); 
					}

					static_cast<std::complex<sFloat> *>(rData)[c0*k0 + c1*k1 + k2] = sm;
printf("%d %+.3lf %+.3lf\n", c0*k0 + c1*k1 + k2, sm.real(), sm.imag());
				}

			return;
		}

		public:

			sFFT  (int d, sFFTtype sType, sFFTdir sDir, bool useMPI, uint Nx, uint Ny, uint Nz, void *data, void *result) : d(d), sType(sType), sDir(sDir), useMPI(useMPI), Nx(Nx), Ny(Ny), Nz(Nz), data(data), result(result), rSize(1<<radix) {

				lx = 0;	for (uint i=Nx; i > 1; i>>=1, lx++);
				ly = 0;	for (uint i=Ny; i > 1; i>>=1, ly++);
				lz = 0;	for (uint i=Nz; i > 1; i>>=1, lz++);
			}

		void	runFFT() {

			iteraFFT (data, result, Nx, 1);
			return;
		}

		void	runButterFFT() {

			uint	cL = rSize;
			uint	cH = Nx >> radix;
			uint	pMax = lx/radix;

			printf("Inputs %u %u\n", 0, Nx);
			butterFFT (data, result, 0, Nx);
			for (uint p = 1; p < pMax; p++) {
				for (uint k0 = 0; k0 < cL; k0++) {
					printf("Inputs %u %u\n", k0, cH);
					void *tttt = malloc(sizeof(std::complex<sFloat>)*Nx);
					memcpy (tttt, result, sizeof(std::complex<sFloat>)*Nx);
					butterFFT (result, tttt, k0, cH);
					memcpy (result, tttt, sizeof(std::complex<sFloat>)*Nx);
					free(tttt);
				}

				cL <<= radix;
				cH >>= radix;
			}

			std::complex<sFloat> *ttt = static_cast<std::complex<sFloat> *>(result);
			for (uint i=0; i<Nx; i++)
				printf("%d %+.3lf %+.3lf\n", i, ttt[i].real(), ttt[i].imag());
			return;
		}
	};

	template<typename sFloat, 4>
	class	sFFT {

		private:

		const uint	rSize;

		int		d;
		uint		Nx, Ny, Nz;
		uint		Gx, Gy, Gz;
		uint		lx, ly, lz;

		void		*data;
		void		*result;

		bool		useMPI;

		sFFTdir		sDir;
		sFFTtype	sType;

		/*	PSEUDOCODE ITERATIVE FFT

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
		void	iteraFFT (void *mData, void *rData, uint M, uint St) {

			if (M == 1) {
				static_cast<std::complex<sFloat> *>(rData)[0] = static_cast<std::complex<sFloat> *>(mData)[0];
				return;
			}

			uint M2 = M >> 1;

			void *fft1 = static_cast<void*>(&static_cast<std::complex<sFloat> *>(mData)[0]);
			void *fft2 = static_cast<void*>(&static_cast<std::complex<sFloat> *>(mData)[St]);

			iteraFFT(fft1, static_cast<void *>(&static_cast<std::complex<sFloat> *>(rData)[0]),  M2, 2*St);
			iteraFFT(fft2, static_cast<void *>(&static_cast<std::complex<sFloat> *>(rData)[M2]), M2, 2*St);


			sFloat iM = ((sFloat) 1.0)/((sFloat) M);

			#pragma omp parallel for schedule(static)
			for (uint i = 0; i < M2; i++) {

				std::complex<sFloat> Wk(cos(twoPi*((sFloat) i)*iM), sin(-twoPi*((sFloat) i)*iM));
				auto	tmp = static_cast<std::complex<sFloat> *>(rData)[i];
				auto	qmp = static_cast<std::complex<sFloat> *>(rData)[i+M2]*Wk;
				static_cast<std::complex<sFloat> *>(rData)[i]		= tmp + qmp;
				static_cast<std::complex<sFloat> *>(rData)[i+M2]	= tmp - qmp;
			}

			return;
		}

		static unsigned int rw(uint k) {

			unsigned char	b0 = b[k >> 0*8 & 0xff];
 			unsigned char	b1 = b[k >> 1*8 & 0xff];
 			unsigned char	b2 = b[k >> 2*8 & 0xff];
 			unsigned char	b3 = b[k >> 3*8 & 0xff];

			return	b0 << 3*8 | b1 << 2*8 | b2 << 1*8 | b3 << 0*8;
		}

		static float r(uint k) {
			return 1./4294967296. * rw(k);
		}

		void	butterFFTLast (void *mData, void *rData, uint u0) { // Only for radix 4

			sFloat	a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i,
					  b1r, b1i, b2r, b2i, b3r, b3i,
				c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i,
				d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i;

			for (int k0 = 0; k0 < u0; ++k0) {

				CommonWeight weight = weights[k0];

				a0r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 0].real();
				a0i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 0].imag();
				a1r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 1].real();
				a1i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 1].imag();
				a2r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 2].real();
				a2i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 2].imag();
				a3r = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 3].real();
				a3i = static_cast<std::complex<sFloat> *>(mData)[4*k0 + 3].imag();

				b1r = - a1i * weight.w1i + a1r;
				b1i = + a1r * weight.w1i + a1i;
				b2r = - a2i * weight.w2i + a2r;
				b2i = + a2r * weight.w2i + a2i;
				b3r = - a3i * weight.w3i + a3r;
				b3i = + a3r * weight.w3i + a3i;
				c0r = + b2r * weight.w2r + a0r;
				c0i = + b2i * weight.w2r + a0i;
				c2r = - b2r * weight.w2r + a0r;
				c2i = - b2i * weight.w2r + a0i;
				c1r = + b3r * weight.w3r + b1r;
				c1i = + b3i * weight.w3r + b1i;
				c3r = - b3r * weight.w3r + b1r;
				c3i = - b3i * weight.w3r + b1i;
				d0r = + c1r * weight.w1r + c0r;
				d0i = + c1i * weight.w1r + c0i;
				d1r = - c1r * weight.w1r + c0r;
				d1i = - c1i * weight.w1r + c0i;
				d2r = - c3i * weight.w1r + c2r;
				d2i = + c3r * weight.w1r + c2i;
				d3r = + c3i * weight.w1r + c2r;
				d3i = - c3r * weight.w1r + c2i;

				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 0].real() = d0r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 0].imag() = d0i;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 1].real() = d1r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 1].imag() = d1i;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 2].real() = d2r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 2].imag() = d2i;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 3].real() = d3r;
				static_cast<std::complex<sFloat> *>(rData)[4*k0 + 3].imag() = d3i;
			}

			return;
		}

		void	butterFFT (void *mData, void *rData, uint k0, uint c0) {

			const uint c1	= c0 >> radix;

			for (uint k2 = 0; k2 < c1; k2++)
				for (uint k1 = 0; k1 < rSize; k1++) {
					std::complex<sFloat> sm(0.,0.);

					for (int j1 = 0; j1 < rSize; j1++) {
						auto arg0 = twoPi*j1*r(k0*rSize);
						auto arg1 = twoPi*j1*r(k1);
						std::complex<sFloat> w0(std::cos(arg0), std::sin(arg0));
						std::complex<sFloat> w1(std::cos(arg1), std::sin(arg1));

						sm +=  w0 * w1 * static_cast<std::complex<sFloat> *>(mData)[c0*k0 + c1*j1 + k2];
auto caca = static_cast<std::complex<sFloat> *>(mData)[c0*k0 + c1*j1 + k2];
printf("%+.3lf %+.3lf | %+.3lf %+.3lf | %+.3lf %+.3lf\n", w0.real(), w0.imag(), w1.real(), w1.imag(), caca.real(), caca.imag()); 
					}

					static_cast<std::complex<sFloat> *>(rData)[c0*k0 + c1*k1 + k2] = sm;
printf("%d %+.3lf %+.3lf\n", c0*k0 + c1*k1 + k2, sm.real(), sm.imag());
				}

			return;
		}

		public:

			sFFT  (int d, sFFTtype sType, sFFTdir sDir, bool useMPI, uint Nx, uint Ny, uint Nz, void *data, void *result) : d(d), sType(sType), sDir(sDir), useMPI(useMPI), Nx(Nx), Ny(Ny), Nz(Nz), data(data), result(result), rSize(1<<radix) {

				lx = 0;	for (uint i=Nx; i > 1; i>>=1, lx++);
				ly = 0;	for (uint i=Ny; i > 1; i>>=1, ly++);
				lz = 0;	for (uint i=Nz; i > 1; i>>=1, lz++);
			}

		void	runFFT() {

			iteraFFT (data, result, Nx, 1);
			return;
		}

		void	runButterFFT() {

			uint	cL = rSize;
			uint	cH = Nx >> radix;
			uint	pMax = lx/radix;

			printf("Inputs %u %u\n", 0, Nx);
			butterFFT (data, result, 0, Nx);
			for (uint p = 1; p < pMax; p++) {
				for (uint k0 = 0; k0 < cL; k0++) {
					printf("Inputs %u %u\n", k0, cH);
					void *tttt = malloc(sizeof(std::complex<sFloat>)*Nx);
					memcpy (tttt, result, sizeof(std::complex<sFloat>)*Nx);
					butterFFT (result, tttt, k0, cH);
					memcpy (result, tttt, sizeof(std::complex<sFloat>)*Nx);
					free(tttt);
				}

				cL <<=  radix;
				cH >>= radix;
			}

			std::complex<sFloat> *ttt = static_cast<std::complex<sFloat> *>(result);
			for (uint i=0; i<Nx; i++)
				printf("%d %+.3lf %+.3lf\n", i, ttt[i].real(), ttt[i].imag());
			return;
		}
	};
}

