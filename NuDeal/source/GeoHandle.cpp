#include "GeoHandle.h"

namespace Geometry
{
	void splittree::Assign(int id) {
		subnode = nullptr; uppernode = nullptr;
		this->id = id; nleaf = 0;
	}

	void splittree::Assign(int id, splittree &uppernode) {
		subnode = nullptr; this->uppernode = &uppernode;
		this->id = id; nleaf = 0;
	}

	void splittree::Branching(int nleaf) {
		subnode = new splittree[nleaf];
		for (int i = 0; i < nleaf; i++) {
			subnode[i].Assign(i, *this);
		}
	}

	void splittree::RecordNodeInfo(vector<int> idvols, vector<double> vol, double3 midpt) {
		thisinfo.idvols = idvols;
		thisinfo.vol = vol;
		thisinfo.midpt = midpt;
	}

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

int GeometryHandler::FindVolId(double3 ptL, double3 ptR, bool lowest) {
	int id = -1;
	
	// Store volumes whose bound boxes cross the finite box
	vector<int> VolInBound;
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
	if (!lowest) {
		for (int i = 0; i < nbnd; i++) {
			vector<double> interpts;
			int idvol = VolInBound[i];
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
				aninter.y = ptR.y; val[0] = ptR.y;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::YZ, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.x = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
			}
			if (mode == ThreeD) {
				aninter.y = ptL.y; aninter.z = ptR.z;
				val[0] = ptL.y; val[1] = ptR.z;
				int ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::YZ, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.x = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
				aninter.y = ptR.y; val[0] = ptR.y;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::YZ, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.x = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
			}


			// Along y axis
			if (mode != OneD) {
				aninter.x = ptL.x; aninter.z = ptL.z;
				val[0] = ptL.x; val[1] = ptL.z;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XZ, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.y = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
				aninter.x = ptR.x; val[0] = ptR.x;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XZ, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.y = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
			}
			if (mode == ThreeD) {
				aninter.x = ptL.x; aninter.z = ptR.z;
				val[0] = ptL.x; val[1] = ptR.z;
				int ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XZ, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.y = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
				aninter.x = ptR.x; val[0] = ptR.x;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XZ, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.y = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
			}


			// Along z axis
			if (mode == ThreeD) {
				aninter.x = ptL.x; aninter.y = ptL.y;
				val[0] = ptL.x; val[1] = ptL.y;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XY, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.z = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
				aninter.x = ptR.x; val[0] = ptR.x;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XY, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.z = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
				aninter.x = ptL.x; aninter.y = ptR.y;
				val[0] = ptL.x; val[1] = ptR.y;
				int ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XY, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.z = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
				aninter.y = ptR.y; val[0] = ptR.y;
				ninter = Volumes[idvol].GetIntersection(UnitSurf::CartPlane::XY, val, interpts);
				for (int j = 0; j < ninter; j++) {
					aninter.z = interpts[j];
					if (InLocalBox(mode, ptL, ptR, aninter)) return id;
				}
			}
		}
	}

	// When alive through above loops, this box has only one volume in it.
	// Find the volume index
	double xmid = 0.5*(ptL.x + ptR.x), ymid = 0.5*(ptL.y + ptR.y), zmid = 0.5*(ptL.z + ptR.z);
	for (int i = 0; i < nbnd; i++) {
		int idvol = VolInBound[i];
		if (Volumes[idvol].IsInside(xmid, ymid, zmid, false)) return idvol;
	}

	// Still alive? Error
	return -100;
}

void GeometryHandler::RecursiveSplit(double3 ptL, double3 ptR, int thisLv, splittree &thisnode) {
	double lx = lx0 / pow3[thisLv] , ly = ly0 / pow3[thisLv], lz = lz0 / pow3[thisLv];
	bool lowest = false;
	lowest = (lx / 3.0 < mintau);
	if (mode != OneD) lowest = (ly / 3.0 < mintau);
	if (mode == ThreeD) lowest = (lz / 3.0 < mintau);

	int idvol = FindVolId(ptL, ptR, lowest);
	if (idvol == -1) {
		thisnode.Branching(27);
		splittree* subnodes = thisnode.GetPtrSubnode();
		double3 ptL1, ptR1;
		lx /= 3.0; ly /= 3.0; lz /= 3.0;
		int subid = 0;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					ptL1.x = ptL.x + (double)i*lx; ptL1.y = ptL.y + (double)j*ly; ptL1.z = ptL.z + (double)k*lz;
					ptL1.x = ptL.x + (double)(i + 1)*lx; ptL1.y = ptL.y + (double)(j + 1)*ly; ptL1.z = ptL.z + (double)(k + 1)*lz;
					RecursiveSplit(ptL1, ptR1, (thisLv + 1), subnodes[subid]);
					subid++;
				}
			}
		}
	}
	else {
		vector<int> ids(idvol);
		vector<double> vol(lx*ly*lz);
		double3 midpt;
		midpt.x = 0.5*(ptL.x + ptR.x); midpt.y = 0.5*(ptL.y + ptR.y); midpt.z = 0.5*(ptL.z + ptR.z);
		thisnode.RecordNodeInfo(ids, vol, midpt);
	}
}

void GeometryHandler::LevelCount(int thisLv, splittree &thisnode) {
	if (thisLv > -1) nnodeLv[thisLv]++;
	splittree *subnode = thisnode.GetPtrSubnode();
	if (subnode != nullptr) {
		int nleaf = thisnode.GetNleaf();
		for (int i = 0; i < nleaf; i++) {
			LevelCount(thisLv + 1, subnode[i]);
		}
	}
}

void GeometryHandler::RecordDiscInfo(int thisLv, splittree &thisnode) {
	int thisLvId = nnodeLv[thisLv], nleaf = thisnode.GetNleaf();
	if (thisLv > 0) {
		nnodeLv[thisLv]++;
		divmap[thisLv][thisLvId] = nnodeLv[thisLv - 1];
		info[thisLv][thisLvId] = thisnode.GetNodeInfo();
	}
	for (int i = 0; i < nleaf; i++) {
		splittree *subnodes = thisnode.GetPtrSubnode();
		RecordDiscInfo(thisLv + 1, subnodes[i]);
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

	// Determine the potentially maximal divlevel
	double log3 = log(3.0);
	int potentdivLv = log(Lx / mintau) / log3;
	if (mode != OneD) potentdivLv = min(potentdivLv, (int)(log(Ly / mintau) / log3));
	if (mode == ThreeD) potentdivLv = min(potentdivLv, (int)(log(Lz / mintau) / log3));
	nnodeLv = new int[potentdivLv]; std::fill_n(nnodeLv, potentdivLv, 0);
	divmap = new int*[potentdivLv];
	info = new subdomain*[potentdivLv];
	for (int i = 0; i < potentdivLv; i++)	pow3.push_back(pow(3.0, i));

	// Ray tracing with the rays parallel to x,y,z axes
	splittree root(0);
	root.Branching(Nx*Ny*Nz);
	splittree *zeronodes = root.GetPtrSubnode();
	double3 ptL, ptR;
	ptL.x = x0; ptL.y = y0; ptL.z = z0;
	ptR.x = x0 + lx0; ptR.y = y0 + ly0; ptR.z = z0 + lz0;
	int inode0 = 0;
	for (int iz = 0; iz < Nz; iz++) {
		for (int iy = 0; iy < Ny; iy++) {
			for (int ix = 0; ix < Nx; ix++) {
				RecursiveSplit(ptL, ptR, 0, zeronodes[inode0]);
				inode0++;
				ptL.x += lx0; ptR.x += lx0;
			}
			ptL.y += ly0; ptR.y += ly0;
		}
		ptL.z += lz0; ptR.z += lz0;
	}

	// Defining nnodeLv and divmap
	LevelCount(-1, root);
	for (int i = 0; i < potentdivLv; i++) {
		if (nnodeLv[i] > 0) {
			divmap[i] = new int[nnodeLv[i]];
			info[i] = new subdomain[nnodeLv[i]];
			std::fill_n(divmap[i], nnodeLv[i], 0);
		}
		else break;
		divlevel++;
	}
	std::fill_n(nnodeLv, divlevel, 0);
	RecordDiscInfo(-1, root);

	// Print-out Division Information
	ofstream DiscOut("Discretization.out");
	for (int i = 0; i < divlevel; i++) {
		for (int j = 0; j < nnodeLv[i]; j++) {
			if (info[i][j].idvols[0] < 0) continue;
			DiscOut << info[i][j].midpt.x << info[i][j].midpt.y << info[i][j].midpt.z << info[i][j].idvols[0] << endl;
		}
	}
	
	return true;
}

}