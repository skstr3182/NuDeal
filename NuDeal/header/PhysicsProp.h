#pragma once
#include "Defines.h"
#include "UnitGeo.h"
#include "XS.h"
#include "GeoHandle.h"

namespace PhysicalDomain {
	enum dir6 {
		xleft, xright,
		yleft, yright,
		zleft, zright
	};

	struct ConnectInfo_t {
		array<int, 6> NeighLv, NeighId, NeighInodes; // 0~6 : xl, xr, yl, yr, zl, zr
		int thisLv, thisId;
	};

	struct CompileInfo_t {
		vector<int> inodes;
		vector<double> weightnodes;
	};
	
	class BaseDomain {
	public:
		using Dimension = Geometry::Dimension;
		using NodeInfo_t = Geometry::NodeInfo_t;
		using DiscInfo_t = Geometry::DiscInfo_t;
		using GeometryHandler = Geometry::GeometryHandler;

	protected:
		bool isalloc = false;
		Dimension mode;
		double x0, y0, z0;
		double lx0, ly0, lz0;

		int Nx, Ny, Nz;
		int nx, ny, nz;
		int nnode, divlevel;
		int *nnodeLv, **upperdivmap, **lowerdivmap;
		int **serialdivmap;

		NodeInfo_t *nodeInfo;
	public:
		BaseDomain() {};

		BaseDomain(const GeometryHandler &rhs) { Create(rhs); }

		void Create(const GeometryHandler &rhs);
	};

	class ConnectedDomain : public BaseDomain {
	protected:
		ConnectInfo_t *connectInfo;

		bool FindNeighbor(dir6 srchdir, array<int,3> ixyz, array<int,2> LvnId, array<int,2> &NeighLvnId);
		void RecursiveConnect(int &ActiveLv, int thisidonset);
	public:
		ConnectedDomain() {};

		ConnectedDomain(const GeometryHandler &rhs) : BaseDomain(rhs) { CreateConnectInfo(); };

		void CreateConnectInfo();
	};

	class CompiledDomain : public BaseDomain {
	private:
		int basetype; // 0 : cartesian boxes, 1: other compiled domains
		CompileInfo_t *compileInfo;
	};

	class RaySegmentDomain : public ConnectedDomain {
	private:
		double *bndflux;
	public:
		RaySegmentDomain() {};

		RaySegmentDomain(const GeometryHandler &rhs) : ConnectedDomain(rhs) {};
	};

	class FlatSrcDomain : public CompiledDomain {
		// for neutronics
	private:
		double *flux0, *flux1, *flux2;
		double *srcF, *srfS, *srcEx;
	};

	class FlatXSDomain : public CompiledDomain {
	private:
		int *idiso;
		double *pnum, *temperature;
		double *xst, *xssm, *xsnf;
	};

	inline void DebugPhysicalDomain() {

		using SurfType = Geometry::UnitSurf::SurfType;
		using CartPlane = Geometry::CartPlane;
		using UnitSurf = Geometry::UnitSurf;
		using UnitVol = Geometry::UnitVol;
		using UnitComp = Geometry::UnitComp;
		using GeometryHandler = Geometry::GeometryHandler;

		double c_cir[3] = { 0.0, 0.0, 0.54 };
		UnitSurf Circle(SurfType::CIRCLE, c_cir, CartPlane::XY);

		double c_zpln0[2] = { -1.0, 0.0 };
		double c_zpln1[2] = { 1.0, 3.0 };
		UnitSurf Zpln0(SurfType::ZPLN, c_zpln0), Zpln1(SurfType::ZPLN, c_zpln1);

		double c_xpln0[2] = { -1.0, -0.63 };
		double c_xpln1[2] = { 1.0, 0.63 };
		double c_ypln0[2] = { -1.0, -0.63 };
		double c_ypln1[2] = { 1.0, 0.63 };
		UnitSurf Xpln0(SurfType::XPLN, c_xpln0), Xpln1(SurfType::XPLN, c_xpln1);
		UnitSurf Ypln0(SurfType::YPLN, c_ypln0), Ypln1(SurfType::YPLN, c_ypln1);

		UnitVol Cylinder;
		Cylinder.Append(Circle);
		Cylinder.Append(Zpln0);
		Cylinder.Append(Zpln1);
		Cylinder.Finalize();

		UnitVol Box;
		Box.Append(Zpln0); Box.Append(Zpln1);
		Box.Append(Ypln0); Box.Append(Ypln1);
		Box.Append(Xpln0); Box.Append(Xpln1);
		Box.Finalize();

		bool isbounded;
		double2 xs, ys, zs;
		isbounded = Cylinder.GetBoundBox(xs, ys, zs);
		cout << "Bound info, (" << isbounded << ")" << endl;
		if (isbounded) cout << "X : [" << xs.x << "," << xs.y << "] ";
		if (isbounded) cout << "Y : [" << ys.x << "," << ys.y << "] ";
		if (isbounded) cout << "Z : [" << zs.x << "," << zs.y << "] " << endl;
		if (isbounded) cout << "Vol = " << Cylinder.GetVolume() << "cm^3" << endl;

		isbounded = Box.GetBoundBox(xs, ys, zs);
		cout << "Bound info, (" << isbounded << ")" << endl;
		if (isbounded) cout << "X : [" << xs.x << "," << xs.y << "] ";
		if (isbounded) cout << "Y : [" << ys.x << "," << ys.y << "] ";
		if (isbounded) cout << "Z : [" << zs.x << "," << zs.y << "] " << endl;
		if (isbounded) cout << "Vol = " << Box.GetVolume() << "cm^3" << endl;

		UnitComp BoxComp(0, Box), CylComp(1, Cylinder);
		CylComp.Finalize();
		GeometryHandler GeoHandle;
		double origin[3] = { -0.63,-0.63,0 }, L[3] = { 1.26, 1.26,3 };
		GeoHandle.SetOrdinates(origin, L);
		GeoHandle.append(BoxComp);
		GeoHandle.append(CylComp);
		GeoHandle.FinalizeVolumes();
		GeoHandle.Discretize(Geometry::Dimension::TwoD, 0.05, 0.005, 6);

		RaySegmentDomain RaySP(GeoHandle);
	}
};