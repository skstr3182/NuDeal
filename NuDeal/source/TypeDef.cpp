#pragma once
#include "TypeDef.h"

void UnitSurf::Transform(double invTM[4][4]) {
	double coeffs[10] = { c_xs, c_ys, c_zs, c_xy, c_yz, c_xz, c_x, c_y, c_z, c };
	double coeffs1[10];
	CalCoeffs(coeffs, invTM, coeffs1);
	c_xs = coeffs1[0]; c_ys = coeffs1[1]; c_zs = coeffs1[2];
	c_xy = coeffs1[3]; c_yz = coeffs1[4]; c_xz = coeffs1[5];
	c_x = coeffs1[6]; c_y = coeffs1[7]; c_z = coeffs1[8]; c = coeffs1[9];
}

UnitSurf::UnitSurf(int surftype, double *_coeff, int cartesianPln) {
	this->Create(surftype, _coeff, cartesianPln);
}

bool UnitSurf::GetPointVar2(int axis, double var[2], double &sol) {
	int i0 = axis, i1, i2;
	double coeff[10] = { c_xs, c_ys, c_zs, c_xy, c_yz, c_xz, c_x, c_y, c_z, c };
	switch (axis) {
	case X:
		i1 = 1; i2 = 2; break;
	case Y:
		i1 = 2; i2 = 0; break;
	case Z:
		i1 = 0; i2 = 1; break;
	}
	double a, b, c;
	a = coeff[i0]; b = coeff[3 + i0] * var[i1] + coeff[3 + i2] * var[i2] + coeff[6 + i0];
	c = coeff[i1] * var[i1] * var[i1] + coeff[i2] * var[i2] * var[i2] + coeff[3 + i1] * var[i1] * var[i2] +
		coeff[6 + i1] * var[i1] + coeff[6 + i2] * var[i1] + coeff[9];
	if (abs(a) < 1.e-10) {
		if (abs(b) > 1.e-10) {
			sol = - b / c;
			return true;
		}
		else {
			return false;
		}
	}
	else {
		sol = - 0.5 * b / a;
		return true;
	}
}

void UnitSurf::Create(int surftype, double *_coeff, int cartesianPln) {
	alloc = true;
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
	for (int i = 0; i < 6; i++) {
		if (coeff[i] > 1.e-10) isCurve = true;
		if (coeff[i] < -1.e-10) isCurve = true;
	}

	c_xs = coeff[0]; c_xy = coeff[3]; c_x = coeff[6];
	c_ys = coeff[1]; c_yz = coeff[4]; c_y = coeff[7];
	c_zs = coeff[2]; c_xz = coeff[5]; c_z = coeff[8];
	c = coeff[9];
}

bool UnitSurf::GetEquation(double *_coeff) const {
	double coeff[10] = { c_xs,c_ys,c_zs,c_xy,c_yz,c_xz,c_x,c_y,c_z,c };
	for (int i = 0; i < 10; i++) _coeff[i] = coeff[i];
	return isCurve;
}

void UnitSurf::Relocate(int dx, int dy, int dz) {
	double invTM[4][4] = { { 0.0, }, };
	invTM[0][0] = 1.0;
	invTM[1][1] = 1.0;
	invTM[2][2] = 1.0;
	invTM[3][3] = 1.0;
	// Update only the fourth rows
	invTM[0][3] = -dx; invTM[1][3] = -dy; invTM[2][3] = -dz;
	Transform(invTM);
}

void UnitSurf::Rotate(double cos, double sin, int Ax) {
	double invTM[4][4] = { { 0.0, }, };
	invTM[0][0] = 1.0;
	invTM[1][1] = 1.0;
	invTM[2][2] = 1.0;
	invTM[3][3] = 1.0;

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
}

int UnitSurf::GetIntersection(int CartPlane, double *val, double &sol1, double &sol2) {
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
			double dist = sqrt(det) / c1;
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

UnitSurf UnitSurf::operator=(const UnitSurf &asurf) {
	double coeff[10];
	asurf.GetEquation(coeff);
	Create(GENERAL, coeff);
	return *this;
}

bool UnitSurf::IsInside(double x, double y, double z, bool includeOn) {
	double det = (c_xs*x + c_xy * y + c_xz * z + c_x)*x + (c_ys*y + c_yz * z + c_y)*y + (c_zs*z + c_z)*z + c;
	if (abs(det) < 1.e-10) {
		if (includeOn) return true;
		return false;
	}
	return (det < 0);
}

int UnitSurf::GetLocalExSelf(double sol[6][3]) {
	// Based on Lagrange multiplier method
	// G(x) = x & F(x,y,z) = 0;
	// grad(F)=lambda*grad(G)

	// Notice
	// - May not be proper for cones
	// - If a corner point is not defined yet a curve line, an exact point would be
	//  defined later by the curve between this surface and {X = a|Y = a|Z = a}.

	// return three-digit value (XYZ)
	// 0 : none for along the axis
	// 1 : left only, 2 : right only, 3 : both sides

	int code = 0;

	if (isCurve) {
		double gradG[3][4] = {
			{2.0*c_xs, c_xy, c_xz, c_x},
			{c_xy, 2.0*c_ys, c_yz, c_y},
			{c_xz, c_yz, 2.0*c_zs, c_z} 
		};
		double coeff0[10] = { c_xs, c_ys, c_zs, c_xy, c_yz, c_xz, c_x, c_y, c_z, c };
		double coeff1[10];
		for (int i = 0; i < 3; i++) {
			// (i = 0) z = c22[0][0]*x+c22[0][1] & y = c22[1][0]*x+c22[1][1]
			// (i = 1) x = c22[0][0]*y+c22[0][1] & z = c22[1][0]*y+c22[1][1]
			// (i = 2) y = c22[0][0]*z+c22[0][1] & x = c22[1][0]*z+c22[1][1]
			// (i = 0) (j,i1,i2) = (0, y, z), (1, z, y)
			// (i = 1)           = (0, z, x), (1, x, z)
			// (i = 2)           = (0, x, y), (1, y, x)
			double c22[2][2];
			int map[2] = { (i + 1) % 3,(i + 2) % 3 };

			bool iscontd = false;
			for (int j = 0; j < 2; j++) {
				// remove i1's term
				int i1 = map[j], i2 = map[(j+1)%2];
				double c0, c1, c2;
				c0 = gradG[i1][i] * gradG[i2][i1] - gradG[i2][i] * gradG[i1][i1]; // i's term
				c1 = gradG[i1][i2] * gradG[i2][i1] - gradG[i2][i2] * gradG[i1][i1]; //i2's term
				c2 = gradG[i1][3] * gradG[i2][i1] - gradG[i2][3] * gradG[i1][i1]; // constant term
				if (abs(c1) < 1.e-10) iscontd = true; // *** The min or max points cannot be determined.
				c22[j][0] = c0 / c1; c22[j][1] = c2 / c1;
			}
			if (iscontd) continue;

			double xyzc[4][4] = { {0} };
			xyzc[i][i] = 1.;
			xyzc[map[0]][i] = c22[1][0]; xyzc[map[0]][3] = c22[1][1];
			xyzc[map[1]][i] = c22[0][0]; xyzc[map[1]][3] = c22[0][1];
			std::fill(xyzc[3], xyzc[3] + 4, 1.0);
			CalCoeffs(coeff0, xyzc, coeff1);

			double bc[2] = { coeff1[i + 6], coeff1[9] };
			if (abs(coeff1[i]) < 1.e-10) {
				double soli = -bc[1] / bc[0];
				double soli1 = c22[1][0] * soli + c22[1][1], soli2 = c22[0][0] * soli + c22[0][1];
				double invlam = gradG[i][i] * soli + gradG[i][map[0]] * soli1 + gradG[i][map[1]] * soli2;
				if (abs(invlam) < 1.e-10) {
					code += 3 * pow10[2 - i];
					sol[i * 2][i] = sol[i * 2 + 1][i] = soli;
					sol[i * 2][map[0]] = sol[i * 2 + 1][map[0]] = soli1;
					sol[i * 2][map[1]] = sol[i * 2 + 1][map[1]] = soli2;
				}
				else if (invlam > 0) {
					code += 2 * pow10[2 - i];
					sol[i * 2 + 1][i] = soli;
					sol[i * 2 + 1][map[0]] = soli1;
					sol[i * 2 + 1][map[1]] = soli2;
				}
				else {
					code += 1 * pow10[2 - i];
					sol[i * 2][i] = soli;
					sol[i * 2][map[0]] = soli1;
					sol[i * 2][map[1]] = soli2;
				}
				continue;
			}
			bc[0] /= coeff1[i]; bc[1] /= coeff1[i];
			double det = bc[0] * bc[0] - 4.0*bc[1];
			if (det > 1.e-10) {
				code += 3 * pow10[2 - i];
				double soli = 0.5*(-bc[0] - sqrt(det));
				double soli1 = c22[1][0] * soli + c22[1][1], soli2 = c22[0][0] * soli + c22[0][1];
				sol[i * 2][i] = soli; sol[i * 2][map[0]] = soli1; sol[i * 2][map[1]] = soli2;
				soli = 0.5*(-bc[0] + sqrt(det));
				soli1 = c22[1][0] * soli + c22[1][1], soli2 = c22[0][0] * soli + c22[0][1];
				sol[i * 2 + 1][i] = soli; sol[i * 2 + 1][map[0]] = soli1; sol[i * 2 + 1][map[1]] = soli2;
			}
		}
	}
	return code;
}

int UnitSurf::GetLocalExPln(double CoeffPln[4], double sol[6][3]) {
	int code = 0;
	if (isCurve) {
		double coeff0[10] = { c_xs, c_ys, c_zs, c_xy, c_yz, c_xz, c_x, c_y, c_z, c };
		for (int i = 0; i < 3; i++) {
			double xyzc[4][4] = { {0.0} };
			for (int j = 0; j < 4; j++) xyzc[j][j] = 1.0;
			if (abs(CoeffPln[i]) > 1.e-10) {
				double invc = 1. / CoeffPln[i];
				// Cancel i's term, x = ay+bz+c
				for (int j = 0; j < 4; j++) xyzc[i][j] = -CoeffPln[j] * invc;
				xyzc[i][i] = 0.;

				double coeff1[10];
				CalCoeffs(coeff0, xyzc, coeff1);
				double coeff2[6] = { coeff1[(i + 1) % 3],coeff1[(i + 2) % 3],coeff1[3 + (i + 1) % 3],coeff1[6 + (i + 1) % 3],coeff1[6 + (i + 2) % 3],coeff1[9] };

				// Get local extremum for an axis
				double sol1[2][2];
				int code1 = GetLocalExCurve((i + 1) % 3, coeff2, sol1);
				if (code1 % 2 == 1) {
					sol[(i + 1) % 3 * 2][(i + 1) % 3] = sol1[0][0];
					sol[(i + 1) % 3 * 2][(i + 2) % 3] = sol1[0][1];
					this->GetPointVar2[i, sol1[0], sol[(i + 1) % 3 * 2][i]];
				}
				if (code1 > 2) {
					sol[(i + 1) % 3 + 1][(i + 1) % 3] = sol1[1][0];
					sol[(i + 1) % 3 + 1][(i + 2) % 3] = sol1[1][1];
					this->GetPointVar2[i, sol1[1], sol[(i + 1) % 3 * 2 + 1][i]];
				}
				code += pow10[(i + 1) % 3] * code1;
				// If (i+2)%3's term was zero, get another pair
				double sol1[2][2];
				int code1 = GetLocalExCurve((i + 2) % 3, coeff2, sol1);
				if (code1 % 2 == 1) {
					sol[(i + 2) % 3 * 2][(i + 1) % 3] = sol1[0][0];
					sol[(i + 2) % 3 * 2][(i + 2) % 3] = sol1[0][1];
					this->GetPointVar2[i, sol1[0], sol[(i + 2) % 3 * 2][i]];
				}
				if (code1 > 2) {
					sol[(i + 2) % 3 + 1][(i + 1) % 3] = sol1[1][0];
					sol[(i + 2) % 3 + 1][(i + 2) % 3] = sol1[1][1];
					this->GetPointVar2[i, sol1[1], sol[(i + 2) % 3 * 2 + 1][i]];
				}
				code += pow10[(i + 2) % 3] * code1;
			}
		}
	}
	return code;
}

int UnitVol::OneIntersection(int idx, int CartPlane, double *val, double &sol1, double &sol2) {
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
		double bufsol;
		if (i == 0) bufsol = sol1;
		if (i == 1) bufsol = sol2;
		switch (CartPlane) {
		case XY:
			z = bufsol;
			break;
		case YZ:
			x = bufsol;
			break;
		case XZ:
			y = bufsol;
			break;
		}
		for (int j = 0; j < nsurf; j++) {
			if (j == i) continue;
			if (!Surfaces[j].IsInside(x, y, z)) {
				ninter--;
				sol1 = sol2;
			}
		}
	}

	return ninter;
}

void UnitVol::Create(int _nsurf, const UnitSurf *&_Surfaces) {
	alloc = true;
	nsurf = _nsurf;
	Surfaces = new UnitSurf[nsurf];
	for (int i = 0; i < _nsurf; i++) Surfaces[i] = _Surfaces[i];
}

void UnitVol::append(const UnitSurf &asurf) {
	alloc = true;
	nsurf += 1;
	UnitSurf **bufsurf = &Surfaces;
	Surfaces = new UnitSurf[nsurf];
	for (int i = 0; i < nsurf - 1; i++) Surfaces[i] = *bufsurf[i];
	Surfaces[nsurf - 1] = asurf;
}

void UnitVol::Relocate(int dx, int dy, int dz) {
	for (int i = 0; i < nsurf; i++) Surfaces[i].Relocate(dx, dy, dz);
}

void UnitVol::Rotate(double cos, double sin, int Ax) {
	for (int i = 0; i < nsurf; i++) Surfaces[i].Rotate(cos, sin, Ax);
}

bool UnitVol::IsInside(double x, double y, double z, bool includeOn) {
	bool inside = true;
	for (int i = 0; i < nsurf; i++) inside = inside && Surfaces[i].IsInside(x, y, z, includeOn);
	return inside;
}

int UnitVol::GetIntersection(int CartPlane, double *val, double **sol) {
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

void UnitVol::ResidentFilter(int codes, int acode, double localEx[6][3], double corners[6][3]) {
	for (int axi = 0; axi < 3; axi++) {
		int globcode = codes % pow10[3 - axi] / pow10[2 - axi];
		int code0 = acode % pow10[3 - axi] / pow10[2 - axi], code1 = 0;
		if (code0 % 2 == 1) {
			if (IsInside(localEx[axi * 2][0], localEx[axi * 2][1], localEx[axi * 2][2], true)) {
				code1 += 1;
				if (globcode % 2 != 1 || localEx[axi * 2][axi] < corners[axi * 2][axi]) {
					std::copy(localEx[axi * 2][0], localEx[axi * 2][2], corners[axi * 2]);
				}
			}
		}
		if (code0 > 1) {
			if (IsInside(localEx[axi * 2 + 1][0], localEx[axi * 2 + 1][1], localEx[axi * 2 + 1][2], true)) {
				code1 += 2;
				if (globcode < 2 || localEx[axi * 2 + 1][axi] > corners[axi * 2 + 1][axi]) {
					std::copy(localEx[axi * 2][0], localEx[axi * 2][2], corners[axi * 2 + 1]);
				}
			}
		}
	}
}

bool UnitVol::GetBoundBox(double xlr[2], double ylr[2], double zlr[2]) {
	double corners[6][3];
	int codes = 0;

	// Symbol for bound-by, (,)
	// o C : Curved surface
	// o P : Plane surface

	// [1] Curved surface local extremum : (C)
	//     1. Get local extremum
	//     2. Extract what are inside volume
	//     3. Revise codes and corner points
	for (int i = 0; i < nsurf; i++) {
		double localEx[6][3] = { {0.0} };
		int acode = Surfaces[i].GetLocalExSelf(localEx);
		if (acode == 0) continue;
		ResidentFilter(codes, acode, localEx, corners);
	}

	// [2] Curve local extremum between curved surfaces and planes : (C,P)
	//     o Same procedures as above
	for (int i = 0; i < nsurf; i++) {
		double localEx[6][3] = { {0.0} };
		for (int j = 0; j < nsurf; j++) {
			if (i == j) continue;
			double coeffj[10];
			bool isCurve = Surfaces[j].GetEquation(coeffj);
			if (isCurve) continue;
			double coeffPln[4] = { coeffj[6],coeffj[7],coeffj[8],coeffj[9] };
			int acode = Surfaces[i].GetLocalExPln(coeffPln, localEx);
			if (acode == 0) continue;
			ResidentFilter(codes, acode, localEx, corners);
		}
	}

	// [3] Curve local extremum between two curved surfaces : (C,C)
	//     o Same procedures as above
	//     ** Not developed yet. This will be done after C5G7 hard-code progresses.

	// From [1] to [3] steps, the local extremum points are obtained.
	// However, from [4] to [5], only the intersection points can be obtained.
	// It is impossible to tell whether each is left or right boundaries of any axes.
	// Explicit comparisons which is largest or smallest along all directions should be carried out.

	std::queue<double[3]> Inters;

	// [4] Intersection of each curved surface and two planes : (C,P,P)
	//     1. Get a line from a pair of planes attached to each curved surface
	//     2. Find two intersections to the curved surface
	//     3. Follow [1]-2 and [1]-3
	//     ** Not developed either.

	// [5] Intersection of three planes : (P,P,P)
	//     1. Get the intersection points from triple planes
	//     2. Follow [1]-2 and [1]-3
	for (int i = 0; i < nsurf; i++) {
		double coeff0[10];
		double IsCurve = Surfaces[i].GetEquation(coeff0);
		if (IsCurve) continue;
		double coeff0_4[4] = { coeff0[6], coeff0[7], coeff0[8], coeff0[9] };
		for (int j = i + 1; j < nsurf; j++) {
			double coeff1[10];
			IsCurve = Surfaces[j].GetEquation(coeff1);
			if (IsCurve) continue;
			double coeff1_4[4] = { coeff1[6], coeff1[7], coeff1[8], coeff1[9] };
			for(int k = j + 1; k < nsurf; k++) {
				double coeff2[10];
				IsCurve = Surfaces[k].GetEquation(coeff2);
				if (IsCurve) continue;
				double coeff2_4[4] = { coeff2[6], coeff2[7], coeff2[8], coeff2[9] };
				double soltriple[3];
				if (GetTriplePoint(coeff0_4, coeff1_4, coeff2_4, soltriple)) Inters.push(soltriple);
			}
		}
	}

	// Filter residents and determine bounding values
	// The corner points from local extremum are collapsed to the queue, Inters.
	// If a plane composed of three resident points holds all the others, return false
	std::queue<double[3]> Residents;
	for (int i = 0; i < 6; i++) {
		int acode = codes % pow10[3 - i] / pow10[2 - i];
		if (acode % 2 == 1) Residents.push(corners[i]);
		if (acode > 1) Residents.push(corners[i]);
	}
	while (!Inters.empty()) {
		double aninter[3];
		std::copy(Inters.front(), Inters.front() + 3, aninter);
		if (IsInside(aninter[0], aninter[1], aninter[2], true)) {
			Residents.push(aninter);
		}
		Inters.pop();
	}
	if (Residents.empty()) return false;
	double aninter0[3];
	std::copy(Residents.front(), Residents.front() + 3, aninter0);
	double boundval[6] = { aninter0[0], aninter0[0], aninter0[1], aninter0[1], aninter0[2], aninter0[2] };
	int seq = 0;
	double DirVec[3];
	while (!Residents.empty()) {
		double aninter[3];
		std::copy(Residents.front(), Residents.front() + 3, aninter);
		// Boundary values update
		for (int i = 0; i < 3; i++) {
			double vl = boundval[i * 2], vr = boundval[i * 2 + 1], vn = aninter[0];
			boundval[i * 2] = (vl > vn) ? vn : vl;
			boundval[i * 2 + 1] = (vr < vn) ? vn : vr;
		}
		// Boundedness check
		double DirVec1[3] = { aninter[0] - aninter0[0], aninter[1] - aninter0[1], aninter[2] - aninter0[2] };
		double det = DirVec1[0] * DirVec1[0] + DirVec1[1] * DirVec1[1] + DirVec1[2] * DirVec1[2];
		if (det < 1.e-10) {
			switch (seq) {
			case 0:
				std::copy(DirVec1, DirVec1 + 3, DirVec);
				seq = 1;
				break;
			case 1:
				DirVec1[0] = DirVec[1] * DirVec1[2] - DirVec[2] * DirVec1[1];
				DirVec1[1] = DirVec[3] * DirVec1[0] - DirVec[0] * DirVec1[3];
				DirVec1[2] = DirVec[0] * DirVec1[1] - DirVec[1] * DirVec1[0];
				det = DirVec1[0] * DirVec1[0] + DirVec1[1] * DirVec1[1] + DirVec1[2] * DirVec1[2];
				if (det > 1.e-10) {
					std::copy(DirVec1, DirVec1 + 3, DirVec);
					seq = 2;
				}
				break;
			case 2:
				det = DirVec[0] * DirVec1[0] + DirVec[1] * DirVec1[1] + DirVec[2] * DirVec1[2];
				if (abs(det) > 1.e-10) seq = 3;
				break;
			case 3:
				break;
			}
		}
		Residents.pop();
	}
	if (seq < 3) return false;
	xlr[0] = boundval[0]; xlr[1] = boundval[1];
	ylr[0] = boundval[2]; ylr[1] = boundval[3];
	zlr[0] = boundval[4]; zlr[1] = boundval[5];
	return true;
}