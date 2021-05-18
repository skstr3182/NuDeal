#pragma once
#include "Defines.h"

enum XSType {
	TOT,
	ABS,
	FIS,
	SCAT,
	Nu,
	Kappa,
	CAP,
	N2N,
	N3N
};

struct XSData {
	bool typeXS[9];
	double **XSval;
	double *ScatKernel, *FisKernel;
};

class XSLib {
private:
	bool isMicLib;
	int niso, nset, ng;
	std::list<XSData> XS_set;

	// Compilation of XS sets
	bool *typeXS;
	double *XSval;
	double *ScatKernel, *FisKernel;

	void init() { niso = nset = ng = 0; }

	void append(XSData datum) { XS_set.push_back(datum); }
public:
	XSLib() { init(); }

	XSLib(bool isMicLib, int ng) { init(); this->ng = ng; this->isMicLib = isMicLib; }
	
	XSLib(bool isMicLib, int ng, int nset);

	XSLib(ifstream libfile);

	void UploadXSData(double ***XS, double ***XSSM);

	void finalize();
};