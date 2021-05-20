#pragma once
#include "Defines.h"
#include "UnitGeo.h"

namespace Geometry {

constexpr double mintau_default = 0.01;
constexpr double maxtau_default = 0.0001;
constexpr double eps_dis = 1.e-10;

enum Dimension {
	ThreeD,
	TwoD,
	OneD
};

struct subdomain {
	double3 midpt;
	vector<int> idvols;
	vector<double> vol;
};

class splittree {
private :
	splittree *subnode, *uppernode;
	int id, nleaf;
	subdomain thisinfo;

public:
	splittree() {}

	splittree(int id) { Assign(id); }

	splittree(int id, splittree &uppernode) { Assign(id, uppernode); }

	void Assign(int id);

	void Assign(int id, splittree &uppernode);

	void Branching(int nleaf);

	void RecordNodeInfo(vector<int> idvols, vector<double> vol, double3 midpt);

	splittree* GetPtrSubnode() { return subnode; }

	int GetId() { return id; }

	int GetNleaf() { return nleaf; }

	int GetUpperId() { return GetId(); }

	subdomain GetNodeInfo() { return thisinfo; }
};

class GeometryHandler {
private:
	bool issetord, isfinal;
	double x0, y0, z0, Lx, Ly, Lz;

	double mintau, maxtau;
	int mode;

	int nvol, nnode, *nnodeLv; // nnodeLv : Number of nodes at each level, size : [divlevel]
	int Nx, Ny, Nz, **divmap; // divmap : Node indices of one-level higher nodes, size [divlevel][nnodeLv[ir]]
	int divlevel;
	double lx0, ly0, lz0;
	subdomain **info;
	
	UnitVol *Volumes;
	queue<UnitVol> UVbuf;

	int *imat;
	queue<int> matidbuf;

	double3 *BdL, *BdR;
	vector<double> pow3;

	void init();

	int FindVolId(double3 ptL, double3 ptR, bool lowest);

	void RecursiveSplit(double3 ptL, double3 ptR, int thisLv, splittree &thisnode);

	void LevelCount(int thisLv, splittree &thisnode);

	void RecordDiscInfo(int thisLv, splittree &thisnode);
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


inline void DebugGeomHandle() {

	using SurfType = UnitSurf::SurfType;
	using CartPlane = UnitSurf::CartPlane;

	double c_cir[3] = { 0.0, 0.0, 0.54 };
	UnitSurf Circle(SurfType::CIRCLE, c_cir, CartPlane::XY);

	double c_zpln0[2] = { -1.0, 0.0 };
	double c_zpln1[2] = { 1.0, 3.0 };
	UnitSurf Zpln0(SurfType::ZPLN, c_zpln0), Zpln1(SurfType::ZPLN, c_zpln1);

	double c_xpln0[2] = { -1.0, 0.63 };
	double c_xpln1[2] = { 1.0, 0.63 };
	double c_ypln0[2] = { -1.0, 0.63 };
	double c_ypln1[2] = { 1.0, 0.63 };
	UnitSurf Xpln0(SurfType::XPLN, c_xpln0), Xpln1(SurfType::XPLN, c_xpln1);
	UnitSurf Ypln0(SurfType::YPLN, c_ypln0), Ypln1(SurfType::YPLN, c_ypln1);

	UnitVol Cylinder;
	Cylinder.Append(Circle);
	Cylinder.Append(Zpln0);
	Cylinder.Append(Zpln1);
	Cylinder.Finalize();

	UnitVol Box;
	Box.Append(Zpln0); Box.Append(Zpln1);
	Box.Append(Ypln0); Box.Append(Ypln1);
	Box.Append(Xpln0); Box.Append(Xpln1);
	Box.Finalize();

	bool isbounded;
	double2 xs, ys, zs;
	isbounded = Cylinder.GetBoundBox(xs, ys, zs);
	cout << "Bound info, (" << isbounded << ")" << endl;
	if (isbounded) cout << "X : [" << xs.x << "," << xs.y << "] ";
	if (isbounded) cout << "Y : [" << ys.x << "," << ys.y << "] ";
	if (isbounded) cout << "Z : [" << zs.x << "," << zs.y << "] " << endl;
	if (isbounded) cout << "Vol = " << Cylinder.GetVolume() << "cm^3" << endl;

	isbounded = Box.GetBoundBox(xs, ys, zs);
	cout << "Bound info, (" << isbounded << ")" << endl;
	if (isbounded) cout << "X : [" << xs.x << "," << xs.y << "] ";
	if (isbounded) cout << "Y : [" << ys.x << "," << ys.y << "] ";
	if (isbounded) cout << "Z : [" << zs.x << "," << zs.y << "] " << endl;
	if (isbounded) cout << "Vol = " << Box.GetVolume() << "cm^3" << endl;
}

}