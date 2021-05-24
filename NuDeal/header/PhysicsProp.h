#pragma once
#include "Defines.h"
#include "UnitGeo.h"
#include "XS.h"
#include "GeoHandle.h"

class AngularFluxDomain : public Geometry::GeometryHandler {
private:
	double *bndflux;
};

class FlatSrcDomain : public Geometry::GeometryHandler {
// for neutronics
private:
	double *flux0, *flux1, *flux2;
	double *srcF, *srfS, *srcEx;
};

class FlatXSDomain : public Geometry::GeometryHandler {
private:
	int *idiso;
	double *pnum, *temperature;
	double *xst, *xssm, *xsnf;
};