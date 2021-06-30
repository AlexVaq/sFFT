#include<random>
#include<cstring>
#include<chrono>
#include"sFFTrt.h"

#include<fftw3.h>

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
			printf("Cagada %zu %lf\n", i, diff);
			areWeGood = false;
		}
	}

	return	areWeGood;
}

constexpr size_t maxSize = 1048576*1024;

int     main (int argc, char *argv[]) {

	std::chrono::high_resolution_clock::time_point start, end;

	size_t	N = 1024;
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
			if (!strcmp (argv[i], "-f")) {
				useSinglePrecision = true;
			}
		}
	}

	auto sFloat = (useSinglePrecision == true) ? sizeof(float) : sizeof(double);

	void	*data	= malloc(sFloat*N*2);
	void	*result	= malloc(sFloat*N*2);
	void	*test	= malloc(sFloat*N*2);

	std::random_device seed;
	std::mt19937_64 mt64;

	bool	awg = true;

	std::complex<double>	acc(0.,0.);

	if (useSinglePrecision == true) {
		std::uniform_real_distribution<float>  uni(-1.0, 1.0);

		#pragma omp parallel for schedule(static)
		for (size_t i=0; i<N; i++)
			static_cast<std::complex<float> *>(data)[i] = std::complex<float> (uni(mt64), uni(mt64));
		for (size_t i=0; i<N; i++)
			acc += static_cast<std::complex<float> *>(data)[i];
		start	= std::chrono::high_resolution_clock::now();
		runFFT(1, sFFT::sFFT_C2C, sFFT::sFFT_Direct, false, N, 1, 1, data, result, sFFT::sFFT_Single);
		end	= std::chrono::high_resolution_clock::now();

		runStdFFTW(N, data, test, true);

		awg	= compareResults(N, static_cast<std::complex<float> *>(result), static_cast<std::complex<float> *>(test));
	} else {
		std::uniform_real_distribution<double> uni(-1.0, 1.0);

		#pragma omp parallel for schedule(static)
		for (size_t i=0; i<N; i++)
			static_cast<std::complex<double>*>(data)[i] = std::complex<double>(uni(mt64), uni(mt64));
		for (size_t i=0; i<N; i++)
			acc += static_cast<std::complex<double>*>(data)[i];
		start	= std::chrono::high_resolution_clock::now();
		runFFT(1, sFFT::sFFT_C2C, sFFT::sFFT_Direct, false, N, 1, 1, data, result, sFFT::sFFT_Double);
		end	= std::chrono::high_resolution_clock::now();

		runStdFFTW(N, data, test, false);

		awg	= compareResults(N, static_cast<std::complex<double>*>(result), static_cast<std::complex<double>*>(test));
	}

	printf("Volume %zu took %.6lf miliseconds\n", N, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6);

	if (awg == true)
		printf("Test results are correct\n");
	else
		printf("FFTW and sFFT differ\n");

	auto sFFTres = static_cast<std::complex<double>*>(result)[0];
	auto FFTWres = static_cast<std::complex<double>*>(test)[0];
		
	printf("Zero momentum FFTW %+.3f %+.3f / sFFT %+.3f %+.3f / Expected %+.3f %+.3f\n", FFTWres.real(), FFTWres.imag(), sFFTres.real(), sFFTres.imag(), acc.real(), acc.imag());

	free(data);
	free(result);
	free(test);

	return	0;
}
