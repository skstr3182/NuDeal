#pragma once
#include "Defines.h"
enum Transform {
	RELOC,
	ROTAT,
	SCALE,
	SHAER
};

enum CartAxis {
	X,
	Y,
	Z
};

enum CartPlane {
	XY,
	YZ,
	XZ
};

enum SurfType {
	XPLN,
	YPLN,
	ZPLN,
	PLN,
	CIRCLE,
	ELLIPSE,
	SPHERE,
	ELLIPSOID,
	GENERAL
};

inline void CalCoeffs(double coeff0[10], double xyzc[4][4], double coeff1[10]) {
	// xyzc : (x', y', z', 1) = g(x,y,z)
	// Convert f(x,y,z) to f(x',y',z') when (x',y',z') = g(x,y,z)
	double c160[4][4], c161[4][4] = { {0} };
	c160[0][0] = coeff0[0]; c160[1][1] = coeff0[1]; c160[2][2] = coeff0[2]; c160[3][3] = 1.0;
	c160[0][1] = coeff0[3]; c160[0][3] = coeff0[6];
	c160[1][2] = coeff0[4]; c160[1][3] = coeff0[7];
	c160[0][2] = coeff0[5]; c160[2][3] = coeff0[8];	
	for (int i = 0; i < 4; i++) {
		for (int j = i; j < 4; j++) {
			for (int ir = 0; ir < 4; ir++) {
				for (int ic = 0; ic < 4; ic++) {
					c161[ir][ic] += c160[i][j] * xyzc[i][ir] * xyzc[j][ic];
				}
			}
		}
	}
	coeff1[0] = c161[0][0]; coeff1[1] = c161[1][1]; coeff1[2] = c161[2][2]; coeff1[9] = c161[3][3];
	coeff1[3] = c161[0][1] + c161[1][0]; coeff1[6] = c161[0][3] + c161[3][0];
	coeff1[4] = c161[0][2] + c161[2][0]; coeff1[7] = c161[1][3] + c161[3][1];
	coeff1[5] = c161[1][2] + c161[2][1]; coeff1[8] = c161[2][3] + c161[3][2];
}

class UnitSurf {
private:
	bool alloc;
	// The realm defined with a surface would be,
	// (c_xs*x+c_x)*x + (c_ys*y+c_y)*y + (c_zs*z+c_z)*z - c < 0
	bool isCurve;
	double c_xs, c_ys, c_zs;
	double c_xy, c_yz, c_xz;
	double c_x, c_y, c_z, c;

	void Transform(double invTM[4][4]);

public:

	UnitSurf() { alloc = false;	}

	UnitSurf(int surftype, double *_coeff, int cartesianPln = XY);

	UnitSurf(const UnitSurf &asurf) { alloc = true;	this->operator=(asurf);	}

	void Create(int surftype, double *_coeff, int cartesianPln = XY);
	
	void Destroy() { alloc = false; }

	bool IsAlloc() { return alloc; }

	bool GetEquation(double *_coeff) const;

	void Relocate(int dx, int dy, int dz);

	void Rotate(double cos, double sin, int Ax);

	int GetIntersection(int CartPlane, double *val, double &sol1, double &sol2);

	UnitSurf operator=(const UnitSurf &asurf);

	bool IsInside(double x, double y, double z);

	int GetLocalMinMax(int cartax, double so1[6][3]);
};

class UnitVol {
private:
	bool alloc;
	int nsurf;
	UnitSurf *Surfaces;

	UnitVol() {	alloc = false; }

	int OneIntersection(int idx, int CartPlane, double *val, double &sol1, double &sol2);

public:

	void Create(int _nsurf, const UnitSurf *&_Surfaces);

	void Destroy() { delete Surfaces; alloc = false; }

	bool IsAlloc() { return alloc; }

	void append(const UnitSurf &asurf);

	void Relocate(int dx, int dy, int dz);

	void Rotate(double cos, double sin, int Ax);

	bool IsInside(double x, double y, double z);

	int GetIntersection(int CartPlane, double *val, double **sol);

	bool GetBoundBox(double xlr[2], double ylr[2], double zlr[2]);
};