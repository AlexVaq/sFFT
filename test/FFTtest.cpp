#include<random>
#include<cstring>
#include<chrono>
#include"sFFTrt.h"

#include<fftw3.h>

#include<array>
	typedef unsigned int uint;
        constexpr std::array<unsigned char, 256> b = {   0, 128, 64, 192, 32, 160,  96, 224, 16, 144, 80, 208, 48, 176, 112, 240,  8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
                                                         4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
                                                         2, 130, 66, 194, 34, 162,  98, 226, 18, 146, 82, 210, 50, 178, 114, 242, 10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
                                                         6, 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246, 14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
                                                         1, 129, 65, 193, 33, 161,  97, 225, 17, 145, 81, 209, 49, 177, 113, 241,  9, 137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
                                                         5, 133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245, 13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
                                                         3, 131, 67, 195, 35, 163,  99, 227, 19, 147, 83, 211, 51, 179, 115, 243, 11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
                                                         7, 135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247, 15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255 };
                static unsigned int rw(uint k) {

                        unsigned char   b0 = b[k >> 0*8 & 0xff];
                        unsigned char   b1 = b[k >> 1*8 & 0xff];
                        unsigned char   b2 = b[k >> 2*8 & 0xff];
                        unsigned char   b3 = b[k >> 3*8 & 0xff];

                        return  b0 << 3*8 | b1 << 2*8 | b2 << 1*8 | b3 << 0*8;
                }

                static float r(uint k) {
                        return 1./4294967296. * rw(k);
                }


void	runStdFFTW(size_t N, void *dataIn, void *dataOut, bool singlePrec) {

	if (singlePrec == true) {
		fftwf_complex	*in  = static_cast<fftwf_complex *>(dataIn);
		fftwf_complex	*out = static_cast<fftwf_complex *>(dataOut);
		fftwf_plan	myPlan;

		auto strPl	= std::chrono::high_resolution_clock::now();
		myPlan		= fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		auto endPl	= std::chrono::high_resolution_clock::now();
		auto plTime	= std::chrono::duration_cast<std::chrono::nanoseconds>(endPl - strPl).count()*1e-6;

		auto strCp	= std::chrono::high_resolution_clock::now();
		fftwf_execute(myPlan);
		auto endCp	= std::chrono::high_resolution_clock::now();
		auto cpTime	= std::chrono::duration_cast<std::chrono::nanoseconds>(endCp - strCp).count()*1e-6;

		fftwf_destroy_plan(myPlan);
		printf("FFTW3 took %.6lf miliseconds to create a plan and %.6lf miliseconds to execute it\n", N, plTime, cpTime);
	} else {
		fftw_complex	*in  = static_cast<fftw_complex *>(dataIn);
		fftw_complex	*out = static_cast<fftw_complex *>(dataOut);
		fftw_plan	myPlan;

		auto strPl	= std::chrono::high_resolution_clock::now();
		myPlan		= fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
		auto endPl	= std::chrono::high_resolution_clock::now();
		auto plTime	= std::chrono::duration_cast<std::chrono::nanoseconds>(endPl - strPl).count()*1e-6;

		auto strCp	= std::chrono::high_resolution_clock::now();
		fftw_execute(myPlan);
		auto endCp	= std::chrono::high_resolution_clock::now();
		auto cpTime	= std::chrono::duration_cast<std::chrono::nanoseconds>(endCp - strCp).count()*1e-6;

		fftw_destroy_plan(myPlan);
		printf("FFTW3 took %.6lf miliseconds to create a plan and %.6lf miliseconds to execute it\n", N, plTime, cpTime);
	}
}

template<typename sFloat>
bool	compareResults (size_t N, std::complex<sFloat> *res1, std::complex<sFloat> *res2) {

	bool	areWeGood = true;
	double	cutoff	  = std::is_same<sFloat,float>::value ? 1e-5 : 1e-10;

	for (size_t i = 0; i<N; i++) {

		auto norm = std::max(std::abs(res1[i]), std::abs(res2[i]));

		if (norm <= cutoff) norm = 1.0;

		auto diff = std::abs(res1[i] - res2[i])/norm;

		if (diff > cutoff) {
			printf("Cagada %zu %lf (%+.3f %+.3f vs %+.3f %+.3f)\n", i, diff, res1[i].real(), res1[i].imag(), res2[i].real(), res2[i].imag());
			areWeGood = false;
		}
	}

	return	areWeGood;
}

constexpr size_t maxSize = 1048576*1024;

int     main (int argc, char *argv[]) {

	std::chrono::high_resolution_clock::time_point start, end;
	double	sFFTtime = 0.;
	double	FFTWtime = 0.;
	double	sFFTbtme = 0.;

	size_t	N = 1024;
	int	radix = 4;
	bool	useSinglePrecision = false;

	if (argc > 1) {
		for (int i=1; i<argc; i++) {
			if (!strcmp (argv[i], "-N")) {
				if (argc == i-1) {
					printf("-N must be followed by a number\n");
					return	1;
				}
				N = atol(argv[i+1]);
				if (N > maxSize) {
					printf("Limits exceeded\n");
					return	1;
				}
				i++;
				continue;
			}

			if (!strcmp (argv[i], "-r")) {
				if (argc == i-1) {
					printf("-r must be followed by 2, 4, 8 or 16\n");
					return	1;
				}
				radix = atol(argv[i+1]);
				if ((radix != 2) && (radix != 4) && (radix != 8) && (radix != 16)) {
					printf("Wrong radix value\n");
					return	1;
				}
				i++;
				continue;
			}

			if (!strcmp (argv[i], "-f")) {
				useSinglePrecision = true;
			}
		}
	}

	auto sFloat = (useSinglePrecision == true) ? sizeof(float) : sizeof(double);

	size_t	bytes	= sFloat*N*2;

	void	*data	= malloc(bytes);
	void	*res1	= malloc(bytes);
	void	*res2	= malloc(bytes);
	void	*test	= malloc(bytes);

	std::random_device seed;
	std::mt19937_64 mt64;

	bool	awg1 = true;
	bool	awg2 = true;

	std::complex<double>	acc(0.,0.);
	std::complex<double>	sFFTres(0.,0.);
	std::complex<double>	sFFTbrs(0.,0.);
	std::complex<double>	FFTWres(0.,0.);

	if (useSinglePrecision == true) {
		std::uniform_real_distribution<float>  uni(-1.0, 1.0);

		for (size_t i=0; i<N; i++)
			static_cast<std::complex<float> *>(data)[i] = std::complex<float> (1.0, 0.0);//uni(mt64), uni(mt64));
		for (size_t i=0; i<N; i++)
			acc += static_cast<std::complex<float> *>(data)[i];

		memset (res1, 0, bytes);
		start	= std::chrono::high_resolution_clock::now();
		runFFT(1, sFFT::sFFT_C2C, sFFT::sFFT_Direct, false, N, 1, 1, data, res1, sFFT::sFFT_Single);
		end	= std::chrono::high_resolution_clock::now();

		sFFTtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6;

		memset (res2, 0, bytes);
		start	= std::chrono::high_resolution_clock::now();
		runButterFFT(1, sFFT::sFFT_C2C, sFFT::sFFT_Direct, false, N, 1, 1, data, res2, sFFT::sFFT_Single, radix);
		end	= std::chrono::high_resolution_clock::now();

		sFFTbtme = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6;

		runStdFFTW(N, data, test, true);

		awg1	= compareResults(N, static_cast<std::complex<float> *>(res1), static_cast<std::complex<float> *>(test));
		awg2	= compareResults(N, static_cast<std::complex<float> *>(res2), static_cast<std::complex<float> *>(test));

		sFFTres	= static_cast<std::complex<float> *>(res1)[0];
		sFFTbrs	= static_cast<std::complex<float> *>(res2)[0];
		FFTWres	= static_cast<std::complex<float> *>(test)[0];
	} else {
		std::uniform_real_distribution<double> uni(-1.0, 1.0);

		for (size_t i=0; i<N; i++)
			static_cast<std::complex<double>*>(data)[i] = std::complex<double>(1.0, 0.0);//uni(mt64), uni(mt64));
		for (size_t i=0; i<N; i++)
			acc += static_cast<std::complex<double>*>(data)[i];

		memset (res1, 0, bytes);
		start	= std::chrono::high_resolution_clock::now();
		runFFT(1, sFFT::sFFT_C2C, sFFT::sFFT_Direct, false, N, 1, 1, data, res1, sFFT::sFFT_Double);
		end	= std::chrono::high_resolution_clock::now();

		sFFTtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6;

		memset (res2, 0, bytes);
		start	= std::chrono::high_resolution_clock::now();
		runButterFFT(1, sFFT::sFFT_C2C, sFFT::sFFT_Direct, false, N, 1, 1, data, res2, sFFT::sFFT_Double, radix);
		end	= std::chrono::high_resolution_clock::now();

		sFFTbtme = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6;

		runStdFFTW(N, data, test, false);

		awg1	= compareResults(N, static_cast<std::complex<double>*>(res1), static_cast<std::complex<double>*>(test));
		awg2	= compareResults(N, static_cast<std::complex<double>*>(res2), static_cast<std::complex<double>*>(test));

		sFFTres	= static_cast<std::complex<double>*>(res1)[0];
		sFFTbrs	= static_cast<std::complex<double>*>(res2)[0];
		FFTWres	= static_cast<std::complex<double>*>(test)[0];
	}

	printf("Volume %zu took %.6lf ms (itera) %.6lf ms (butterfly)\n", N, sFFTtime, sFFTbtme);

	if ((awg1 == true) && (awg2 == true))
		printf("Test results are correct\n");
	else
		printf("FFTW and sFFT differ\n");

	printf("Zero momentum FFTW %+.3f %+.3f / sFFT %+.3f %+.3f / sFFTb %+.3f %+.3f / Expected %+.3f %+.3f\n", FFTWres.real(), FFTWres.imag(), sFFTres.real(), sFFTres.imag(), sFFTbrs.real(), sFFTbrs.imag(), acc.real(), acc.imag());

	free(data);
	free(res1);
	free(res2);
	free(test);

	for (uint i=0; i<16; i++)
		printf("%u - r(%u) = %+.3f - r(rSize*%u) = %+.3f\n", i, i, r(i), i, r((1<<radix)*i));

	return	0;
}
