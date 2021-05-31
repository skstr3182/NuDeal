#include "PhysicsProp.h"

namespace PhysicalDomain {
	void BaseDomain::Create(const GeometryHandler &rhs) {
		DiscInfo_t mesg;
		rhs.GetDiscretizationInfo(mesg);
		mode = mesg.mode;
		x0 = mesg.x0; y0 = mesg.y0; z0 = mesg.z0;
		lx0 = mesg.lx0; ly0 = mesg.ly0; lz0 = mesg.lz0;
		Nx = mesg.Nx; Ny = mesg.Ny; Nz = mesg.Nz;
		nx = 3; ny = (mode != Dimension::OneD) ? 3 : 1; nz = (mode == Dimension::ThreeD) ? 3 : 1;
		nnode = mesg.nnode; divlevel = mesg.divlevel;
		nnodeLv = new int[divlevel]; upperdivmap = new int*[divlevel]; lowerdivmap = new int*[divlevel - 1];
		serialdivmap = new int*[divlevel]; nodeInfo = new NodeInfo_t[nnode];

		int inode = 0;
		for (int i = 0; i < divlevel; i++) {
			nnodeLv[i] = mesg.nnodeLv[i];
			upperdivmap[i] = new int[nnodeLv[i]];
			serialdivmap[i] = new int[nnodeLv[i]];
			std::copy(mesg.upperdivmap[i], mesg.upperdivmap[i] + nnodeLv[i], upperdivmap[i]);
			std::fill(serialdivmap[i], serialdivmap[i] + nnodeLv[i], -1);
			for (int j = 0; j < nnodeLv[i]; j++) {
				if (mesg.info[i][j].idvols[0] < 0) continue;
				serialdivmap[i][j] = inode;
				nodeInfo[inode] = mesg.info[i][j];
				inode++;
			}
			if (i > 0) {
				int i0 = i - 1;
				lowerdivmap[i0] = new int[nnodeLv[i0] + 1];
				std::fill(lowerdivmap[i0], lowerdivmap[i0] + nnodeLv[i0] + 1, 0);
				for (int j = 0; j < nnodeLv[i]; j++) {
					int jnode = upperdivmap[i][j];
					lowerdivmap[i0][jnode + 1]++;
				}
				for (int j = 0; j < nnodeLv[i0]; j++) {
					lowerdivmap[i0][j + 1] += lowerdivmap[i0][j];
				}
			}
		}

		isalloc = true;
	}

	bool ConnectedDomain::FindNeighbor(dir6 srchdir, array<int, 3> ixyz, array<int, 2> LvnId, array<int, 2> &NeighLvnId) {
		int thisLv = LvnId[0], thisId = LvnId[1];
		int ix = ixyz[0], iy = ixyz[1], iz = ixyz[2];
		int mx, my, mz;
		int jx = ix, jy = iy, jz = iz;

		if (thisLv == 0) { mx = Nx; my = Ny; mz = Nz; }
		else { mx = nx; my = ny; mz = nz; }

		// Search upward if an neighboring is not in the local box
		int upid = upperdivmap[thisLv][thisId], nowid = thisId, nowLv = thisLv;
		stack<int> jxs, jys, jzs;
		int jdoff;
		jxs.push(jx); jys.push(jy); jzs.push(jz);
		while (true) {
			bool upfurther;
			switch (srchdir) {
			case xleft:
				upfurther = (jx == 0);
				break;
			case yleft:
				upfurther = (jy == 0);
				break;
			case zleft:
				upfurther = (jz == 0);
				break;
			case xright:
				upfurther = (jx + 1 == mx);
				break;
			case yright:
				upfurther = (jy + 1 == my);
				break;
			case zright:
				upfurther = (jz + 1 == mz);
				break;
			default:
				upfurther = false;
				break;
			}
			if (!upfurther) break;
			if (nowLv == 0) return false;
			nowid = upperdivmap[nowLv][nowid];
			nowLv--;
			int jdoff = nowid;
			if (nowLv == 0) { mx = Nx; my = Ny; mz = Nz; }
			else {
				upid = upperdivmap[nowLv][nowid];
				jdoff -= lowerdivmap[nowLv - 1][upid];
			}
			jx = jdoff % mx; jy = jdoff / mx % my; jz = jdoff / mx / my;
			jxs.push(jx); jys.push(jy); jzs.push(jz);
		}
		// Search downward while moving the indices[jx,jy,jz] a step until it reaches to the same level of box
		while (!jxs.empty()) {
			jx = jxs.top(); jy = jys.top(); jz = jzs.top();
			jxs.pop(); jys.pop(); jzs.pop();
			int jxoff = 0, jyoff = 0, jzoff = 0;
			if (nowLv > 0) { mx = nx; my = ny; mz = nz; jxoff = mx; jyoff = my; jzoff = mz; }
			switch (srchdir) {
			case xleft:
				jx = (jx + jxoff - 1) % mx; break;
			case yleft:
				jy = (jy + jyoff - 1) % my; break;
			case zleft:
				jz = (jz + jzoff - 1) % mz; break;
			case xright:
				jx = (jx + 1) % mx; break;
			case yright:
				jy = (jy + 1) % my; break;
			case zright:
				jz = (jz + 1) % mz; break;
			}
			jdoff = jx + mx * (jy + my * jz);
			if (nowLv > 0) jdoff += lowerdivmap[nowLv - 1][upid];
			upid = jdoff;
			nowLv++;
			if (nowLv < divlevel) if (lowerdivmap[nowLv - 1][upid] == lowerdivmap[nowLv - 1][upid + 1]) break;
		}
		NeighLvnId[0] = nowLv - 1; NeighLvnId[1] = jdoff;
	}

	void ConnectedDomain::RecursiveConnect(int &ActiveLv, int thisidonset) {
		int mx, my, mz;
		int lowid0 = 0, thisid = thisidonset;
		if (ActiveLv < divlevel - 1) lowid0 = lowerdivmap[ActiveLv][thisid];
		if (ActiveLv == 0) { mx = Nx; my = Ny; mz = Nz;	}
		else { mx = nx; my = ny; mz = nz; }
		for (int iz = 0; iz < mz; iz++) {
			for (int iy = 0; iy < my; iy++) {
				for (int ix = 0; ix < mx; ix++) {
					//if (ActiveLv == 0) cout << "Connecting [ " << ix << ',' << iy << ',' << iz << "] ..." << endl;
					int lowid1 = (ActiveLv < divlevel - 1) ? lowerdivmap[ActiveLv][thisid + 1] : 0;
					if (lowid0 != lowid1) {
						ActiveLv++;
						RecursiveConnect(ActiveLv, lowid0);
					}
					else {
						int inode = serialdivmap[ActiveLv][thisid];
						array<int, 3> ixyz = { ix,iy,iz }; array<int, 2> LvnId = { ActiveLv,thisid };
						array<int, 2> NeighLvnId;
						std::fill(connectInfo[inode].NeighLv.begin(), connectInfo[inode].NeighLv.end(), -1);
						std::fill(connectInfo[inode].NeighId.begin(), connectInfo[inode].NeighId.end(), -1);
						std::fill(connectInfo[inode].NeighInodes.begin(), connectInfo[inode].NeighInodes.end(), -1);
						if (FindNeighbor(xleft, ixyz, LvnId, NeighLvnId)) { 
							connectInfo[inode].NeighLv[0] = NeighLvnId[0];
							connectInfo[inode].NeighId[0] = NeighLvnId[1];
							connectInfo[inode].NeighInodes[0] = serialdivmap[NeighLvnId[0]][NeighLvnId[1]];
						}
						if (FindNeighbor(xright, ixyz, LvnId, NeighLvnId)) {
							connectInfo[inode].NeighLv[1] = NeighLvnId[0];
							connectInfo[inode].NeighId[1] = NeighLvnId[1];
							connectInfo[inode].NeighInodes[1] = serialdivmap[NeighLvnId[0]][NeighLvnId[1]];
						}
						if (FindNeighbor(yleft, ixyz, LvnId, NeighLvnId)) {
							connectInfo[inode].NeighLv[2] = NeighLvnId[0];
							connectInfo[inode].NeighId[2] = NeighLvnId[1];
							connectInfo[inode].NeighInodes[2] = serialdivmap[NeighLvnId[0]][NeighLvnId[1]];
						}
						if (FindNeighbor(yright, ixyz, LvnId, NeighLvnId)) {
							connectInfo[inode].NeighLv[3] = NeighLvnId[0];
							connectInfo[inode].NeighId[3] = NeighLvnId[1];
							connectInfo[inode].NeighInodes[3] = serialdivmap[NeighLvnId[0]][NeighLvnId[1]];
						}
						if (FindNeighbor(zleft, ixyz, LvnId, NeighLvnId)) {
							connectInfo[inode].NeighLv[4] = NeighLvnId[0];
							connectInfo[inode].NeighId[4] = NeighLvnId[1];
							connectInfo[inode].NeighInodes[4] = serialdivmap[NeighLvnId[0]][NeighLvnId[1]];
						}
						if (FindNeighbor(zright, ixyz, LvnId, NeighLvnId)) {
							connectInfo[inode].NeighLv[5] = NeighLvnId[0];
							connectInfo[inode].NeighId[5] = NeighLvnId[1];
							connectInfo[inode].NeighInodes[5] = serialdivmap[NeighLvnId[0]][NeighLvnId[1]];
						}
					}
					lowid0 = lowid1;
					thisid++;
				}
			}
		}
		ActiveLv--;
	}

	void ConnectedDomain::CreateConnectInfo(){
		cout << "Enter Create Connect Info" << endl;
		connectInfo = new ConnectInfo_t[nnode];
		for (int i = 0; i < divlevel; i++) {
			for (int j = 0; j < nnodeLv[i]; j++) {
				int inode = serialdivmap[i][j];
				if (inode > -1) {
					connectInfo[inode].thisLv = i;
					connectInfo[inode].thisId = j;
				}
			}
		}
		int activelv = 0;
		RecursiveConnect(activelv, 0);

		cout << "Connection Done!" << endl;
		ofstream ConctOut("Connection.out");
		for (int i = 0; i < nnode; i++) {
			ConctOut << connectInfo[i].thisLv << ' ' << connectInfo[i].thisId << endl;;
			ConctOut << "  xl : " << connectInfo[i].NeighLv[0] << ' ' << connectInfo[i].NeighId[0] << ' ' << connectInfo[i].NeighInodes[0] << endl;
			ConctOut << "  xr : " << connectInfo[i].NeighLv[1] << ' ' << connectInfo[i].NeighId[1] << ' ' << connectInfo[i].NeighInodes[1] << endl;
			ConctOut << "  yl : " << connectInfo[i].NeighLv[2] << ' ' << connectInfo[i].NeighId[2] << ' ' << connectInfo[i].NeighInodes[2] << endl;
			ConctOut << "  yr : " << connectInfo[i].NeighLv[3] << ' ' << connectInfo[i].NeighId[3] << ' ' << connectInfo[i].NeighInodes[3] << endl;
			ConctOut << "  zl : " << connectInfo[i].NeighLv[4] << ' ' << connectInfo[i].NeighId[4] << ' ' << connectInfo[i].NeighInodes[4] << endl;
			ConctOut << "  zr : " << connectInfo[i].NeighLv[5] << ' ' << connectInfo[i].NeighId[5] << ' ' << connectInfo[i].NeighInodes[5] << endl;
		}
		ofstream ConctOut2("SimpleConnection.out");
		for (int i = 0; i < nnode; i++) {
			for (int j = 0; j < 6; j++) {
				if (connectInfo[i].NeighInodes[j] < 0) continue;
				ConctOut2 << i << ' ' << connectInfo[i].NeighInodes[j] << endl;
			}
		}
	}
}