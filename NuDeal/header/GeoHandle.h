#pragma once
#include "Defines.h"
#include "UnitGeo.h"


namespace Geometry {

constexpr double mintau_default = 0.01;
constexpr double maxtau_default = 0.0001;

enum Dimension {
	ThreeD,
	TwoD,
	OneD
};

class GeometryHandler {
private:
	bool issetord, isfinal;
	double x0, y0, z0, Lx, Ly, Lz;

	double mintau, maxtau;
	int mode;

	int nvol, nnode, *nnodeLv;
	int Nx, Ny, Nz, **divmap;
	int divlevel;
	double lx0, ly0, lz0;
	
	UnitVol *Volumes;
	queue<UnitVol> UVbuf;

	int *imat;
	queue<int> matidbuf;

	double3 *BdL, *BdR;

	void init();

	int FindVolId(double3 pt);
public:
	GeometryHandler() { init(); }

	GeometryHandler(double origin[3], double L[3]);
	
	void SetOrdinates(double origin[3], double L[3]);

	void append(UnitVol &avol, int matid) { UVbuf.push(avol); matidbuf.push(matid); }

	void append(UnitComp &acomp);

	void Create(int ncomp, UnitComp *Comps) { for (int i = 0; i < ncomp; i++) append(Comps[i]); }

	void FinalizeVolumes();

	bool Discretize(int Dim, double minlen = mintau_default, double maxlen = maxtau_default);
};

}