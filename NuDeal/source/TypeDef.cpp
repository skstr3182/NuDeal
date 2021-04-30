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

bool UnitSurf::IsInside(double x, double y, double z) {
	double det = (c_xs*x + c_xy * y + c_xz * z + c_x)*x + (c_ys*y + c_yz * z + c_y)*y + (c_zs*z + c_z)*z + c;
	if (det = 0) return false;
	return (det < 0);
}

int UnitSurf::GetLocalMinMax(int cartax, double sol[6][3]) {
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
					code += 3*pow(10,2-i);
					sol[i * 2][i] = sol[i * 2 + 1][i] = soli;
					sol[i * 2][map[0]] = sol[i * 2 + 1][map[0]] = soli1;
					sol[i * 2][map[1]] = sol[i * 2 + 1][map[1]] = soli2;
				}
				else if (invlam > 0) {
					code += 2 * pow(10, 2 - i);
					sol[i * 2 + 1][i] = soli;
					sol[i * 2 + 1][map[0]] = soli1;
					sol[i * 2 + 1][map[1]] = soli2;
				}
				else {
					code += 1 * pow(10, 2 - i);;
					sol[i * 2][i] = soli;
					sol[i * 2][map[0]] = soli1;
					sol[i * 2][map[1]] = soli2;
				}
				continue;
			}
			bc[0] /= coeff1[i]; bc[1] /= coeff1[i];
			double det = bc[0] * bc[0] - 4.0*bc[1];
			if (det > 1.e-10) {
				code += 3 * pow(10, 2 - i);
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

bool UnitVol::IsInside(double x, double y, double z) {
	bool inside = true;
	for (int i = 0; i < nsurf; i++) inside = inside && Surfaces[i].IsInside(x, y, z);
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

bool UnitVol::GetBoundBox(double xlr[2], double ylr[2], double zlr[2]) {
	int *codes = new int[nsurf];

	for (int i = 0; i < nsurf; i++) {
		// X boundaries

		// Y boundaries

		// Z boundaries

	}
}