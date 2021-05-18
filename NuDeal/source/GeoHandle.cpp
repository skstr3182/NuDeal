#include "GeoHandle.h"

void GeometryHandler::init() {
	nvol = nnode = divlevel = 0; isfinal = false; issetord = false;
	x0 = y0 = z0 = Lx = Ly = Lz = 0.0;
}

int GeometryHandler::FindVolId(double3 pt) {
	int id = -1;
	
	// Store volumes whose bound boxes hold the point.
	vector<int> VolInBound;
	for (int i = 0; i < nvol; i++) {
		if (pt.x<BdL[i].x - eps_geo || pt.x>BdR[i].x + eps_geo) continue;
		if (!mode == OneD) {
			if (pt.y<BdL[i].y - eps_geo || pt.y>BdR[i].y + eps_geo) continue;
			if (mode == ThreeD) {
				if (pt.z<BdL[i].z - eps_geo || pt.z>BdR[i].z + eps_geo) continue;
			}
		}
		VolInBound.push_back(i);
	}

	// First search : Including those on which a point is
	//  1. Call UniVol.IsInside(x,y,z,true)
	//  2. Unless multiple volumes are detected, return id;

	// Second search : Only those in which a point is
	//  1. Call UnitVol.IsInside(x,y,z,false) for ids found at first searches
	//  2. If 
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
	int nvol = acomp.GetNvol();
	const UnitVol* Volumes = acomp.GetVolumes();
	const int* imat = acomp.GetMatIds();
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
		double xs[2], ys[2], zs[2];
		Volumes[i].GetBoundBox(xs, ys, zs);
		BdL[i].x = xs[0]; BdL[i].y = ys[0]; BdL[i].z = zs[0];
		BdR[i].x = xs[1]; BdR[i].y = ys[1]; BdR[i].z = zs[1];
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
		if (!mode == OneD && lym < maxtau) {
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
	if (!mode == OneD) Ny = Ly / minlen;
	if (mode == ThreeD) Nz = Nz / minlen;
	lx0 = Lx / (double)Nx; ly0 = Ly / (double)Ny; lz0 = Lz / (double)Nz;

	// Ray tracing with the rays parallel to x,y,z axes
	int ivol[8];
	
	return true;
}