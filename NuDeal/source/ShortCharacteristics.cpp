#include "ShortCharacteristics.h"

constexpr double PI = 3.14159265358979;

namespace Transport {
void AngularQuadrature::CreateSet(AngQuadType type, vector<int> parameters){
	quadtype = type;
	switch (type)
	{
	case Transport::AngQuadType::UNIFORM:
	{
		int nazi = parameters[0], npolar = parameters[1];
		int nangle_oct = nazi * npolar;

		double3 thisomega;
		double dtheta = PI * 0.25 / nazi, dphi = PI * 0.5 / npolar;
		for (int ip = 0; ip < npolar; ip++) {
			double phi = dphi * 0.5 * static_cast<double>(1 + 2 * ip);
			thisomega.z = cos(phi);
			for (int ia = 0; ia < nazi; ia++) {
				double theta = dtheta * 0.5 * static_cast<double>(1 + 2 * ia);
				thisomega.x = sin(phi) * cos(theta); thisomega.y = sin(phi) * sin(theta);
				omega.push_back(thisomega);
				weights.push_back(dtheta * (cos(phi - dphi * 0.5) - cos(phi + dphi * 0.5)));
			}
		}

		break;
	}
	case Transport::AngQuadType::LS:
		break;
	case Transport::AngQuadType::GC:
		break;
	case Transport::AngQuadType::Bi3:
		break;
	case Transport::AngQuadType::G_Bi3:
		break;
	case Transport::AngQuadType::QR:
		break;
	default:
		break;
	}
}

AsymptoticExp::AsymptoticExp(double tol, double xL, double xR) {
	this->tol = tol; range.x = xL; range.y = xR;
	int npts = (range.y - range.x) / tol + 1;
	double dx = (range.y - range.x) / static_cast<double>(npts);
	for (int i = 0; i < npts + 1; i++) {
		double pointXval = range.x + dx * static_cast<double>(i);
		double pointExp = exp(pointXval);
		PtsXval.push_back(pointXval); PtsExp.push_back(pointExp - 1.0);
	}
}

double AsymptoticExp::ExpFast(double xval) {
	int i = (xval - range.x) / tol;
	double pointExp = PtsExp[i], pointXval = PtsXval[i];
	return (pointExp * (xval - pointXval));
}

double AsymptoticExp::ExpSafe(double xval) {
	if ((xval - range.x) * (xval - range.y) > 0) return exp(xval);
	int i = (xval - range.x) / tol;
	double pointExp = PtsExp[i], pointXval = PtsXval[i];
	return (pointExp * (xval - pointXval));
}

void DriverSCMOC::Initialize(RaySegment& rhs, FlatSrc& FSR, const FlatXS& FXR){
	mode = rhs.GetDimension();
	rhs.GetBaseSizes(Nxyz, nxyz, nnode, divlevel);
	rhs.GetNodeSizes(lxyz0);
	trackL.Create(QuadSet.nangle_oct); //wtIF.resize(3);

	nodeinfo = rhs.GetBaseNodeInfo().data();
	innodeLv = rhs.GetInNodeLv().data();
	scatorder = FXR.GetScatOrder();

	nFXR = FXR.GetNblocks(); nFSR = FSR.GetNblocks();

	vector<double> rpow3;

	double val = 1;
	for (int i = 0; i < divlevel; i++) { rpow3.push_back(val); val /= 3.; }

	double Ax = lxyz0.y * lxyz0.z, Ay = lxyz0.x * lxyz0.z, Az = lxyz0.x * lxyz0.y;
	double Atot = Ax + Ay + Az;

	//wtIF.resize(3); wtIF[0] = Ax / Atot; wtIF[1] = Ay / Atot; wtIF[2] = Az / Atot;
	wtIF.x = Ax / Atot; wtIF.y = Ay / Atot; wtIF.z = Az / Atot;

	vector<double> trackL0(QuadSet.nangle_oct);
	for (int iang = 0; iang < QuadSet.nangle_oct; iang++) {
		double3 omega = QuadSet.omega[iang];
		double3 Ldeproj; // de-projected length
		Ldeproj.x = lxyz0.x / omega.x; Ldeproj.y = lxyz0.y / omega.y; Ldeproj.z = lxyz0.z / omega.z;

		double Lmin = min({ Ldeproj.x, Ldeproj.y, Ldeproj.z });
		double3 Lproj, Lrem;
		Lproj.x = Lmin * omega.x; Lproj.y = Lmin * omega.y; Lproj.z = Lmin * omega.z;
		Lrem.x = Lmin - Lproj.x; Lrem.y = Lmin - Lproj.y; Lrem.z = Lmin - Lproj.z;

		double A2 = Lproj.x * (Lrem.y + Lrem.z) + Lproj.y * (Lrem.x + Lrem.z) + Lproj.z * (Lrem.x + Lrem.y);
		double A3 = 2.0 * (Lproj.y * Lproj.z + Lproj.x * Lproj.y + Lproj.x * Lproj.z);
		trackL0[iang] = Lmin / Atot * (1 + A2 / 2.0 + A3 / 3.0);
	}

	trackL.Create(QuadSet.nangle_oct,divlevel);
	for (int lv = 0; lv < divlevel; lv++) {
		for (int iang = 0; iang < QuadSet.nangle_oct; iang++) {
			trackL(iang, lv) = trackL0[iang] * rpow3[lv];
		}
	}

	scalarflux = FSR.GetScalarFlux().data();
	srcF = FSR.GetFisSrc().data(); srcS = FSR.GetScatSrc().data();
	xst = FXR.GetTotalXS().data();

	const int *idFSR2FXR = FXR.GetDecompileId().data();
	idFSR = FSR.GetDecompileId().data(); wFSR = FSR.GetDecompileWeights().data();

	for (int i = 0; i < nnode; i++) idFXR.push_back(idFSR2FXR[idFSR[i]]);
}

DriverSCMOC::DriverSCMOC(int ng, AngQuadType quadtype, vector<int> quadparameter) 
	: Exponent(0.0001, -40.0, 0.0) 
{
	this->ng = ng;
	QuadSet.CreateSet(quadtype, quadparameter);
}

void DriverSCMOC::AvgInFlux(double *outx, double *outy, double *outz, double *avgIn) {
	for (int ig = 0; ig < ng; ig++) {
		for (int iang = 0; iang < QuadSet.nangle_oct; iang++) {
			avgIn[ig + iang * ng] = wtIF.x * outx[ig + iang * ng] 
				+ wtIF.y * outy[ig + iang * ng] + wtIF.z * outz[ig + iang * ng];
		}
	}
}

void DriverSCMOC::TrackAnode(int inode, double *angflux, double *src) {
	int ActiveLv = innodeLv[inode];
	int iFXR = idFXR[inode];
	for (int iangle = 0; iangle < QuadSet.nangle_oct; iangle++) {
		for (int ig = 0; ig < ng; ig++) {
			double Len = trackL(iangle, ActiveLv), sigT = xst[ig + iFXR * ng];
			double expTau = Exponent.ExpFast(-Len * sigT);
			double chord = src[ig + iangle * ng] / sigT;
			angflux[ig + iangle * ng] = chord + (angflux[ig + iangle * ng] - chord) * expTau;
			scalarflux[ig + iFXR * ng] += angflux[ig + iangle * ng] * QuadSet.weights[iangle];
			/*if (scatorder >= 1) {
			scalarflux[ig + (iFXR + nFXR) * ng] += 
			scalarflux[ig + (iFXR + 2 * nFXR) * ng] += 
			scalarflux[ig + (iFXR + 3 * nFXR) * ng] += 
			} */
		}
	}
}
}