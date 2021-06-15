#include "XS.h"
#include "Array.h"

namespace XS {
XSLib::XSLib(bool isMicro, int ng, int nset, int scatorder){
	Init();

	this->isMicro = isMicro; this->ng = ng;
	if (isMicro) niso = nset;
	else this->nset = nset;
	this->scatorder = scatorder;

	Resize();
}

void XSLib::UploadXSData(array<bool,10> typeXS, vector<vector<vector<double>>> XS, vector<vector<double>> XSSM) {
	for (int i = 0; i < nset; i++) {
		XS_set[i].typeXS = typeXS;
		int itype = 0;
		for (int j = 0; j < 10; j++) {
			if (typeXS[j]) {
				XS_set[i].XSval[j].Create(ng, ntemp);
				std::copy(XS[i][itype].begin(), XS[i][itype].end(), XS_set[i].XSval[j].begin());
				itype++;
			}
		}
		//for (int j = 0; j < scatorder; j++) {
		//	XS_set[i].XSSM[j].Create(ng, ng, ntemp);
		//}
		XS_set[i].XSSM[0].Create(ng, ng, ntemp);
		std::copy(XSSM[i].begin(), XSSM[i].end(), XS_set[i].XSSM[0].begin());
	}
}
}