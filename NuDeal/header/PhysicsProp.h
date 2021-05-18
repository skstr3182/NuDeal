#pragma once
#include "Defines.h"
#include "UnitGeo.h"
#include "XS.h"
#include "GeoHandle.h"

class FluxDomain : public Geometry::GeometryHandler {
// for neutronics
private:
	double *flux;
};

class XSDomain : public Geometry::GeometryHandler {
private:
	int *idiso;
	double *pnum, *temperature;
};