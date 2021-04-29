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

	void Transform(double **invTM) {
		double coeffs[10] = { c_xs, c_ys, c_zs, c_xy, c_yz, c_xz, c_x, c_y, c_z, c };
		const int map[4][4] = { {0, 3, 5, 6}, {3, 1, 4, 7}, {5, 7, 2, 8}, {6, 7, 8, 9} };
		double trans[4][4];
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				trans[i][j] = 0.;
			}
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				double coeff = coeffs[map[i][j]];
				// i-th row and j-th col in invTM multiply
				for (int k = 0; k < 4; k++) {
					for (int l = 0; l < 4; l++) {
						trans[k][l] += coeff * invTM[i][k] * invTM[l][j];
					}
				}
			}
		}
		c_xs = trans[0][0]; c_ys = trans[1][1]; c_zs = trans[2][2]; c = trans[3][3];
		c_xy = trans[0][1] + trans[1][0]; c_yz = trans[1][2] + trans[2][1]; c_xy = trans[0][2] + trans[2][0];
		c_x = trans[0][3] + trans[3][0]; c_y = trans[1][3] + trans[3][1]; c_z = trans[2][3] + trans[3][2];
	}
public:
	UnitSurf() {
		alloc = false;
	}

	UnitSurf(int surftype, double *_coeff, bool _inout, int cartesianPln = XY) {
		alloc = true;
		inout = _inout;
		double coeff[10];
		for (int i = 0; i < 10; i++) coeff[i] = 0.;
		switch (surftype) {
		case XPLN:
			coeff[5] = _coeff[0];
			coeff[9] = -_coeff[1];
			break;
		case YPLN:
			coeff[6] = _coeff[0];
			coeff[9] = -_coeff[1];
			break;
		case ZPLN:
			coeff[7] = _coeff[0];
			coeff[9] = -_coeff[1];
			break;
		case PLN:
			coeff[5] = _coeff[0];
			coeff[6] = _coeff[1];
			coeff[7] = _coeff[2];
			coeff[9] = -_coeff[3];
			break;
		case CIRCLE:
			switch (cartesianPln) {
			case XY:
				coeff[0] = 1.; coeff[1] = 1.;
				coeff[6] = -2. * _coeff[0]; coeff[7] = -2.*_coeff[1];
				coeff[9] = _coeff[0] * _coeff[0] + _coeff[1] * _coeff[1] - coeff[2] * coeff[2];
				break;
			case YZ:
				coeff[1] = 1.; coeff[2] = 1.;
				coeff[7] = -2. * _coeff[0]; coeff[8] = -2.*_coeff[1];
				coeff[9] = _coeff[0] * _coeff[0] + _coeff[1] * _coeff[1] - coeff[2] * coeff[2];
				break;
			case XZ:
				coeff[0] = 1.; coeff[2] = 1.;
				coeff[6] = -2. * _coeff[0]; coeff[8] = -2.*_coeff[1];
				coeff[9] = _coeff[0] * _coeff[0] + _coeff[1] * _coeff[1] - coeff[2] * coeff[2];
				break;
			}
			break;
		default: // GENERAL
			for (int i = 0; i < 10; i++) coeff[i] = _coeff[i];
			break;
		}

		isCurve = false;
		for (int i = 0; i < 5; i++) {
			if (coeff[i] > 1.e-10) isCurve = true;
			if (coeff[i] < -1.e-10) isCurve = true;
		}

		c_xs = coeff[0]; c_xy = coeff[3]; c_x = coeff[6];
		c_ys = coeff[1]; c_yz = coeff[4]; c_y = coeff[7];
		c_zs = coeff[2]; c_xz = coeff[5]; c_z = coeff[8];
		c = coeff[9];
	}

	UnitSurf(const UnitSurf &asurf) {
		alloc = true;
		this->operator=(asurf);
	}

	void Create(int surftype, double *_coeff, bool _inout, int cartesianPln=XY) {
		inout = _inout;
		double coeff[10];
		for (int i = 0; i < 10; i++) coeff[i] = 0.;
		switch (surftype) {
		case XPLN:
			coeff[5] = _coeff[0];
			coeff[9] = -_coeff[1];
			break;
		case YPLN:
			coeff[6] = _coeff[0];
			coeff[9] = -_coeff[1];
			break;
		case ZPLN:
			coeff[7] = _coeff[0];
			coeff[9] = -_coeff[1];
			break;
		case PLN:
			coeff[5] = _coeff[0];
			coeff[6] = _coeff[1];
			coeff[7] = _coeff[2];
			coeff[9] = -_coeff[3];
			break;
		case CIRCLE:
			switch (cartesianPln) {
			case XY:
				coeff[0] = 1.; coeff[1] = 1.;
				coeff[6] = -2. * _coeff[0]; coeff[7] = -2.*_coeff[1];
				coeff[9] = _coeff[0] * _coeff[0] + _coeff[1] * _coeff[1] - coeff[2] * coeff[2];
				break;
			case YZ:
				coeff[1] = 1.; coeff[2] = 1.;
				coeff[7] = -2. * _coeff[0]; coeff[8] = -2.*_coeff[1];
				coeff[9] = _coeff[0] * _coeff[0] + _coeff[1] * _coeff[1] - coeff[2] * coeff[2];
				break;
			case XZ:
				coeff[0] = 1.; coeff[2] = 1.;
				coeff[6] = -2. * _coeff[0]; coeff[8] = -2.*_coeff[1];
				coeff[9] = _coeff[0] * _coeff[0] + _coeff[1] * _coeff[1] - coeff[2] * coeff[2];
				break;
			}
			break;
		default: // GENERAL
			for (int i = 0; i < 10; i++) coeff[i] = _coeff[i];
			break;
		}
		
		isCurve = false;
		for (int i = 0; i < 5; i++) {
			if (coeff[i] > 1.e-10) isCurve = true;
			if (coeff[i] < -1.e-10) isCurve = true;
		}

		c_xs = coeff[0]; c_xy = coeff[3]; c_x = coeff[6];
		c_ys = coeff[1]; c_yz = coeff[4]; c_y = coeff[7];
		c_zs = coeff[2]; c_xz = coeff[5]; c_z = coeff[8];
		c = coeff[9];
		if (alloc) {
			eyeTr = true;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					TransM[i][j] = 0.; invTM[i][j] = 0.;
				}
				TransM[i][i] = 1.; invTM[i][i] = 1.;
			}
		}
	}

	void Destroy() { alloc = false; }

	bool IsAlloc() { return alloc; }

	bool GetEquation(double *_coeff) const {
		double coeff[10] = { c_xs,c_ys,c_zs,c_xy,c_yz,c_xz,c_x,c_y,c_z,c };
		for (int i = 0; i < 10; i++) _coeff[i] = coeff[i];
		return inout;
	}

	void Relocate(int dx, int dy, int dz) {
		double **invTM = new double*[4];
		for (int i = 0; i < 4; i++) {
			invTM[i] = new double[4];
			for (int j = 0; j < 4; j++) {
				invTM[i][j] = 0.;
			}
			invTM[i][i] = 1.;
		}
		// Update only the fourth rows
		invTM[0][3] = -dx; invTM[1][3] = -dy; invTM[2][3] = -dz;
		Transform(invTM);
		for (int i = 0; i < 4; i++) {
			delete invTM[i];
		}
		delete invTM;
	}

	void Rotate(double cos, double sin, int Ax) {
		double **invTM = new double*[4];
		for (int i = 0; i < 4; i++) {
			invTM[i] = new double[4];
			for (int j = 0; j < 4; j++) {
				invTM[i][j] = 0.;
			}
			invTM[i][i] = 1.;
		}
		int ir1, ir2;
		switch (Ax) {
		case X:
			ir1 = 1; ir2 = 2;
			break;
		case Y:
			ir1 = 2; ir2 = 0;
			break;
		case Z:
			ir1 = 0; ir2 = 1;
			break;
		default:
			break;
		}
		invTM[ir1][ir1] = cos; invTM[ir2][ir2] = cos;
		invTM[ir1][ir2] = sin; invTM[ir2][ir1] = -sin;
		Transform(invTM);
		for (int i = 0; i < 4; i++) {
			delete invTM[i];
		}
		delete invTM;
	}
	
	int GetIntersection (int CartPlane, double *val, double &sol1, double &sol2) {
		double c2, c1, c0;
		if (isCurve) {
			switch (CartPlane) {
			case XY:
				c2 = c_zs;
				c1 = c_xz * val[0] + c_yz * val[1];
				c0 = c_xs * val[0] * val[0] + c_ys * val[1] * val[1] + c_xy * val[0] * val[1] +
					c_x * val[0] + c_y * val[1] + c;
				break;
			case YZ:
				c2 = c_xs;
				c1 = c_xy * val[0] + c_xz * val[1];
				c0 = c_ys * val[0] * val[0] + c_zs * val[1] * val[1] + c_yz * val[0] * val[1] +
					c_y * val[0] + c_z * val[1] + c;
				break;
			case XZ:
				c2 = c_ys;
				c1 = c_xy * val[0] + c_yz * val[1];
				c0 = c_xs * val[0] * val[0] + c_zs * val[1] * val[1] + c_xz * val[0] * val[1] +
					c_x * val[0] + c_z * val[1] + c;
				break;
			default:
				break;
			}
			double det = c1 * c1 - 4.*c2*c0;
			if (det < 0.) {
				return 0;
			}
			else {
				double dist = sqrt(det)/c1;
				sol1 = (-c1 / c1 - dist) * 0.5;
				sol2 = (-c1 / c1 + dist) * 0.5;
				if (dist < 1.e-12) {
					return 1;
				}
				else {
					return 2;
				}
			}
		}
		else {
			switch (CartPlane) {
			case XY:
				sol1 = -(c_x*val[0] + c_y * val[1] + c);
				break;
			case YZ:
				sol1 = -(c_x*val[0] + c_y * val[1] + c);
				break;
			case XZ:
				sol1 = -(c_x*val[0] + c_y * val[1] + c);
				break;
			default:
				break;
			}
			return 1;
		}
	}

	inline UnitSurf operator=(const UnitSurf &asurf) {
		double coeff[10];
		bool inout;
		inout = asurf.GetEquation(coeff);
		Create(GENERAL, coeff, inout);
	}

	bool IsInside(double x, double y, double z) {
		double det = (c_xs*x + c_xy * y + c_xz * z + c_x)*x + (c_ys*y + c_yz * z + c_y)*y + (c_zs*z + c_z)*z + c;
		if (det = 0) return false;
		return ((inout) == (det > 0));
	}
};

class UnitVol {
private:
	bool alloc;
	int nsurf;
	UnitSurf *Surfaces;

	UnitVol() {
		alloc = false;
	}

	int OneIntersection(int idx, int CartPlane, double *val, double &sol1, double &sol2) {
		double x, y, z;
		switch (CartPlane) {
		case XY:
			x = val[0]; y = val[1];
			break;
		case YZ:
			y = val[0]; z = val[1];
			break;
		case XZ:
			x = val[0]; z = val[1];
			break;
		}
		int ninter = Surfaces[idx].GetIntersection(CartPlane, val, sol1, sol2);
		int minter = ninter;
		for (int i = 0; i < minter; i++) {
			switch (CartPlane) {
			case XY:
				z = sol[i];
				break;
			case YZ:
				x = sol[i];
				break;
			case XZ:
				y = sol[i];
				break;
			}
			for (int j = 0; j < nsurf; j++) {
				if (j == i) continue;
				if (!Surfaces[j].IsInside(x,y,z)) {
					ninter--;
					sol1 = sol2;
				}
			}
		}

		return ninter;
	}
public:
	void Create(int _nsurf, const UnitSurf *&_Surfaces) {
		alloc = true;
		nsurf = _nsurf;
		Surfaces = new UnitSurf[nsurf];
		for (int i = 0; i < _nsurf; i++) Surfaces[i] = _Surfaces[i];
	}

	void Destroy() { delete Surfaces; alloc = false; }

	bool IsAlloc() { return alloc; }

	void append(const UnitSurf &asurf) {
		true;
		nsurf += 1;
		UnitSurf **bufsurf = &Surfaces;
		Surfaces = new UnitSurf[nsurf];
		for (int i = 0; i < nsurf - 1; i++) Surfaces[i] = *bufsurf[i];
		Surfaces[nsurf - 1] = asurf;
	}

	void Relocate(int dx, int dy, int dz) {
		for (int i = 0; i < nsurf; i++) Surfaces[i].Relocate;
	}

	void Rotate(double cos, double sin, int Ax) {
		for (int i = 0; i < nsurf; i++) Surfaces[i].Rotate;
	}

	bool IsInside(double x, double y, double z) {
		bool inside=true;
		for (int i = 0; i < nsurf; i++) inside = (inside&&Surfaces[i].IsInside);
		return inside;
	}

	int GetIntersection(int CartPlane, double *val, double **sol) {
		double **bufs = new double*[nsurf];
		for (int i = 0; i < nsurf; i++) bufs[i] = new double[2];
		int *ninters = new int[nsurf];

		int ntotint = 0;
		for (int i = 0; i < nsurf; i++) {
			ninters[i] = OneIntersection(i, CartPlane, val, bufs[i][0], bufs[i][1]);
			ntotint += ninters[i];
		}

		*sol = new double[ntotint];
		ntotint = 0;
		for (int i = 0; i < nsurf; i++) {
			for (int j = 0; j < ninters[i]; j++) {
				*sol[ntotint] = bufs[i][j];
				ntotint++;
			}
			delete bufs[i];
		}
		delete bufs; delete ninters;

		return ntotint;
	} 
};