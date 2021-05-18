#include "GeoHandle.h"

namespace Geometry
{

inline bool InLocalBox(int mode, double3 ptL, double3 ptR, double3 aninter) {
	if (aninter.x > ptL.x + eps_geo && aninter.x < ptR.x - eps_geo) {
		if (mode == OneD) return true;
		else {
			if (aninter.y > ptL.y + eps_geo && aninter.y < ptR.y - eps_geo) {
				if (mode == TwoD) return true;
				else {
					if (aninter.z > ptL.z + eps_geo && aninter.z < ptR.z - eps_geo) {
						return true;
					}
				}
			}
		}
	}
	return false;
}

void GeometryHandler::init() {
	nvol = nnode = divlevel = 0; isfinal = false; issetord = false;
	x0 = y0 = z0 = Lx = Ly = Lz = 0.0;
}

int GeometryHandler::FindVolId(double3 ptL, double3 ptR) {
	int id = -1;
	
	// Store volumes whose bound boxes cross the finite box
	deque<int> VolInBound;
	for (int i = 0; i < nvol; i++) {
		int inboundcount = 0;
		double OL, OR;
		double Ll, Lr, Rl, Rr;
		double Ldet, Rdet;
		OL = BdL[i].x; OR = BdR[i].x;
		Ll = OL - ptL.x; Lr = OL - ptR.x;
		Rl = OR - ptL.x; Rr = OR - ptR.x;
		if (Ll*Lr < 0 && Rl*Rr < 0) inboundcount++;
		if (Lr*Rr > 0 && Ll*Rl > 0) continue;
		if (mode != OneD) {
			OL = BdL[i].y; OR = BdR[i].y;
			Ll = OL - ptL.y; Lr = OL - ptR.y;
			Rl = OR - ptL.y; Rr = OR - ptR.y;
			if (Ll*Lr < 0 && Rl*Rr < 0) inboundcount++;
			if (Lr*Rr > 0 && Ll*Rl > 0) continue;
			if (mode == ThreeD) {
				OL = BdL[i].z; OR = BdR[i].z;
				Ll = OL - ptL.z; Lr = OL - ptR.z;
				Rl = OR - ptL.z; Rr = OR - ptR.z;
				if (Ll*Lr < 0 && Rl*Rr < 0) inboundcount++;
				if (Lr*Rr > 0 && Ll*Rl > 0) continue;
				if (inboundcount == 3) return id;
			}
			else {
				if (inboundcount == 2) return id;
			}
		}
		else {
			if (inboundcount == 1) return id;
		}
		VolInBound.push_back(i);
	}

	int nbnd = VolInBound.size();
	for (int i = 0; i < nbnd; i++) {
		vector<double> interpts;
		int idvol = VolInBound.front();
		array<double, 2> val;

		// Along x axis
		double3 aninter;
		aninter.y = ptL.y; aninter.z = ptL.z;
		val[0] = ptL.y; val[1] = ptL.z;
		int ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::YZ, val, interpts);
		for (int j = 0; j < ninter; j++) {
			aninter.x = interpts[j];
			if (InLocalBox(mode, ptL, ptR, aninter)) return id;
		}
		if (mode != OneD) {

		}
		if (mode == ThreeD) {

		}

		VolInBound.pop_front();
	}
}

GeometryHandler::GeometryHandler(double origin[3], double L[3]) {
	nvol = nnode = divlevel = 0;
	SetOrdinates(origin, L);
}

void GeometryHandler::SetOrdinates(double origin[3], double L[3]) {
	issetord = true;
	x0 = origin[0]; y0 = origin[1]; z0 = origin[2];
	Lx = L[0]; Ly = L[1]; Lz = L[2];	
}

void GeometryHandler::append(UnitComp &acomp) {
	int nvol = acomp.GetNumVolumes();
	const auto& Volumes = acomp.GetVolumes();
	const auto& imat = acomp.GetMatIds();
	for (int i = 0; i < nvol; i++) {
		UVbuf.push(Volumes[i]);
		matidbuf.push(imat[i]);
	}
}

void GeometryHandler::FinalizeVolumes() {
	nvol = UVbuf.size();
	Volumes = new UnitVol[nvol]; imat = new int[nvol];
	BdL = new double3[nvol]; BdR = new double3[nvol];
	for (int i = 0; i < nvol; i++) {
		Volumes[i] = UVbuf.front();
		imat[i] = matidbuf.front();
		UVbuf.pop(); matidbuf.pop();
		double2 xs, ys, zs;
		Volumes[i].GetBoundBox(xs, ys, zs);
		BdL[i].x = xs.x; BdL[i].y = ys.x; BdL[i].z = zs.x;
		BdR[i].x = xs.y; BdR[i].y = ys.y; BdR[i].z = zs.y;
	}
	if (!issetord) {
		x0 = BdL[0].x; y0 = BdL[0].y; z0 = BdL[0].z;
		Lx = BdR[0].x - x0; Ly = BdR[0].y - y0; Lz = BdR[0].z - z0;
		for (int i = 1; i < nvol; i++) {
			double xl = BdL[i].x, yl = BdL[i].y, zl = BdL[i].z;
			x0 = (x0 > xl) ? xl : x0;
			y0 = (y0 > yl) ? yl : y0;
			z0 = (z0 > zl) ? zl : z0;
			Lx = (Lx > BdR[i].x - xl) ? Lx : BdR[i].x - xl;
			Ly = (Ly > BdR[i].y - yl) ? Ly : BdR[i].y - yl;
			Lz = (Lz > BdR[i].z - zl) ? Lz : BdR[i].z - zl;
		}
		issetord = true;
	}
}

bool GeometryHandler::Discretize(int Dim, double minlen, double maxlen) {
	if (!isfinal) return false;
	mode = Dim; mintau = minlen; maxtau = maxlen;
	for (int i = 0; i < nvol; i++) {
		double lxm = BdR[i].x - BdL[i].x, lym = BdR[i].y - BdL[i].y, lzm = BdR[i].z - BdL[i].z;
		if (lxm < maxtau) {
			cout << "  *** Warning: Given maximal length is too small. Reset to " << lxm;
			maxtau = lxm;
		}
		if (mode != OneD && lym < maxtau) {
			cout << "  *** Warning: Given maximal length is too small. Reset to " << lym;
			maxtau = lym;
		}
		if (mode == ThreeD && lzm < maxtau) {
			cout << "  *** Warning: Given maximal length is too small. Reset to " << lzm;
			maxtau = lzm;
		}
	}
	
	// Define zero-level parameters
	Nx = Lx / minlen + 1; Ny = 1; Nz = 1;
	if (mode != OneD) Ny = Ly / minlen;
	if (mode == ThreeD) Nz = Nz / minlen;
	lx0 = Lx / (double)Nx; ly0 = Ly / (double)Ny; lz0 = Lz / (double)Nz;

	// Ray tracing with the rays parallel to x,y,z axes
	int ivol[8];
	
	return true;
}

}