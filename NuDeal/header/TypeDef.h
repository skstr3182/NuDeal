#pragma once
#include <cmath>
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

class UnitSurf {
private:
	bool alloc;
	// The realm defined with a surface would be,
	// If (inout),  ((c_xs*x+c_x)*x + (c_ys*y+c_y)*y + (c_zs*z+c_z)*z - c < 0
	// If (!inout),  ((c_xs*x+c_x)*x + (c_ys*y+c_y)*y + (c_zs*z+c_z)*z - c > 0
	bool isCurve, inout;
	double c_xs, c_ys, c_zs;
	double c_xy, c_yz, c_xz;
	double c_x, c_y, c_z, c;

	void Transform(double **invTM);
public:
	UnitSurf() { alloc = false;	}

	UnitSurf(int surftype, double *_coeff, bool _inout, int cartesianPln = XY);

	UnitSurf(const UnitSurf &asurf) { alloc = true;	this->operator=(asurf);	}

	void Create(int surftype, double *_coeff, bool _inout, int cartesianPln = XY);
	
	void Destroy() { alloc = false; }

	bool IsAlloc() { return alloc; }

	bool GetEquation(double *_coeff) const;

	void Relocate(int dx, int dy, int dz);

	void Rotate(double cos, double sin, int Ax);

	int GetIntersection(int CartPlane, double *val, double &sol1, double &sol2);

	UnitSurf operator=(const UnitSurf &asurf);

	bool IsInside(double x, double y, double z);
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
};