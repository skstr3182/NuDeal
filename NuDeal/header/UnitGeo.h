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

const int pow10[4] = { 1, 10, 100, 1000 };

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

inline int GetLocalExCurve(int axis, double coeff[6], double sol[2][2]) {
	// Based on Lagrange multiplier method
	// G(x) = x & F(x,y) = 0;
	// grad(F)=lambda*grad(G)

	// return single digit value
	// 0 : none for along the axis
	// 1 : left only, 2 : right only, 3 : both sides
	double gradG[2][3] = {
		{2.0*coeff[0], coeff[2], coeff[3]},
		{coeff[2], 2.0*coeff[1], coeff[4]}
	};
	int i0 = axis, i1 = (axis + 1) % 2;
	if (abs(gradG[i1][i1]) < 1.e-10) return 0;
	double c1 = -gradG[i1][i0] / gradG[i1][i1], c0 = -gradG[i1][2] / gradG[i1][i1];
	double a, b, c;
	a = coeff[i0] + coeff[2] * c1 + coeff[i1] * c1 * c1; // [i0][i0] + [i0][i1] + [i1][i1]
	b = coeff[2 + i0] + coeff[2] * c0 + 2.0 * coeff[i1] * c0 * c1 + coeff[2 + i1] * c1; // [i0] + [i0][i1] + [i1]*[i1] + [i1]
	c = coeff[i1] * c0 * c0 + coeff[2 + i1] * c0 + coeff[5]; // [i1][i1] + [i1] + constant
	if (abs(a) < 1.e-10) return 0;
	else {
		b /= a; c /= a;
		double det = b * b - 4.0 * c;
		if (abs(det) < 1.e-10) {
			double soli0 = -b * 0.5;
			double soli1 = c1 * soli0 + c0;
			sol[0][i0] = sol[1][i0] = soli0; sol[0][i1] = sol[1][i1] = soli1;
			double lambda = gradG[i0][i0] * soli0 + gradG[i0][i1] * soli1 + gradG[i0][2];
			if (lambda < 0) return 1;
			else return 2;
		}
		else if (det > 0) {
			double soli0 = 0.5*(-b - sqrt(det));
			double soli1 = c1 * soli0 + c0;
			sol[0][i0] = soli0; sol[0][i1] = soli1;
			soli0 = 0.5*(-b + sqrt(det));
			soli1 = c1 * soli0 + c0;
			sol[1][i0] = soli0; sol[1][i1] = soli1;
			return 3;
		}
	}
}

inline bool GetTriplePoint(double coeff0[4], double coeff1[4], double coeff2[4], double sol[3]) {
	// Construct (x, y)^T = A * (a, b) with combinations of {EQ1,EQ2} and {EQ1,EQ3}
	// If A is singular, return false
	double A[2][2];
	A[0][0] = coeff2[2] * coeff0[0] - coeff0[2] * coeff2[0];
	A[0][1] = coeff2[2] * coeff0[1] - coeff0[2] * coeff2[1];
	A[1][0] = coeff2[2] * coeff1[0] - coeff1[2] * coeff2[0];
	A[1][1] = coeff2[2] * coeff1[1] - coeff1[2] * coeff2[1];
	double det = A[0][0] * A[1][1] - A[1][0] * A[0][1];
	if (abs(det) < 1.e-10) return false;
	double a = coeff2[2] * coeff0[3] - coeff0[2] * coeff2[3];
	double b = coeff2[2] * coeff1[3] - coeff1[2] * coeff2[3];
	a /= det; b /= det;
	sol[0] = A[1][1] * a - A[0][1] * b;
	sol[1] = A[0][0] * b - A[1][0] * a;
	if (abs(coeff0[2]) > 1.e-10) {
		sol[2] = -(coeff0[0] * sol[0] + coeff0[1] * sol[1] + coeff0[3]) / coeff0[2];
		return true;
	}
	if (abs(coeff1[2]) > 1.e-10) {
		sol[2] = -(coeff1[0] * sol[0] + coeff1[1] * sol[1] + coeff1[3]) / coeff1[2];
		return true;
	}
	if (abs(coeff2[2]) > 1.e-10) {
		sol[2] = -(coeff2[0] * sol[0] + coeff2[1] * sol[1] + coeff2[3]) / coeff2[2];
		return true;
	}
	return false; // when det is not small enough
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

	bool GetPointVar2(int axis, double var[2], double &sol);
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

	bool IsInside(double x, double y, double z, bool includeOn = false);

	int GetLocalExSelf(double so1[6][3]);

	int GetLocalExPln(double CoeffPln[4], double sol[6][3]);
};

class UnitVol {
private:
	bool alloc;
	int nsurf;
	UnitSurf *Surfaces;

	UnitVol() {	alloc = false; }

	int OneIntersection(int idx, int CartPlane, double *val, double &sol1, double &sol2);

	void ResidentFilter(int codes, int acode, double localEx[6][3], double corners[6][3]);
public:
	void Create(int _nsurf, const UnitSurf *&_Surfaces);

	void Destroy() { delete Surfaces; alloc = false; }

	bool IsAlloc() { return alloc; }

	void append(const UnitSurf &asurf);

	void Relocate(int dx, int dy, int dz);

	void Rotate(double cos, double sin, int Ax);

	bool IsInside(double x, double y, double z, bool includeOn = false);

	int GetIntersection(int CartPlane, double *val, double **sol);

	bool GetBoundBox(double xlr[2], double ylr[2], double zlr[2]);
};