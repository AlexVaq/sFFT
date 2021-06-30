namespace	sFFT {

	typedef	enum	sFFTtype {
		sFFT_C2C,
		sFFT_R2R,
		sFFT_C2R,
		sFFT_R2C
	}	sFFTtype;

	typedef	enum	sFFTdir {
		sFFT_Direct,
		sFFT_Inverse
	}	sFFTdir;

	typedef	enum	sFFTprec {
		sFFT_Single = 4,
		sFFT_Double = 8
	}	sFFTprec;
}

