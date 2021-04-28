#pragma once
enum Transform {
	RELOC,
	ROTAT,
	SCALE,
	SHAER
};

enum RotAxis {
	X,
	Y,
	Z
};

enum RayLine {
	XY,
	YZ,
	XZ
};

class UnitVol {
private:
	double c_xs, c_ys, c_zs;
	double c_xy, c_yz, c_xz;
	double c_x, c_y, c_z, c;
	double TransM[4][4], invTM[4][4];

	UnitVol() {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				TransM[i][j] = 0.; invTM[i][j] = 0.;
			}
			TransM[i][i] = 1.; invTM[i][i] = 1.;
		}
	}

	void Relocate(int dx, int dy, int dz) {
		// Update only the fourth rows
		for (int ic = 0; ic < 4; ic++) {
			TransM[3][ic] += dx * TransM[0][ic] + dy * TransM[1][ic] + dz * TransM[2][ic];
			invTM[3][ic] += (-dx) * invTM[0][ic] -dy * invTM[1][ic] - dz * invTM[2][ic];
		}
	}

	void Rotate(double cos, double sin, RotAxis Ax) {
		int ir1, ir2;
		switch (Ax) {
		case X:
			ir1 = 1; ir2 = 2;
			break;
		case Y:
			ir1 = 0; ir2 = 2;
			break;
		case Z:
			ir1 = 0; ir2 = 1;
			break;
		default:
			break;
		}
		for (int ic = 0; ic < 3; ic++) {
			TransM[ir1][ic] = cos * TransM[ir1][ic] - sin * TransM[ir2][ic];
			TransM[ir2][ic] = sin * TransM[ir1][ic] + cos * TransM[ir2][ic];
			invTM[ir1][ic] = cos * invTM[ir1][ic] + sin * invTM[ir2][ic];
			invTM[ir2][ic] = (-sin) * invTM[ir1][ic] + cos * invTM[ir2][ic];
		}
	}
public:
	void GetEquation(int *_coeff) {
		int coeff[10] = { c_xs,c_ys,c_zs,c_xy,c_yz,c_xz,c_x,c_y,c_z,c };
		for (int i = 0; i < 10; i++) _coeff[i] = coeff[i];
	}

};