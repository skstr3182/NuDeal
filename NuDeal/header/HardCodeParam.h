#pragma once
#include "Defines.h"

const double asypitch = 21.42;
const int pina = 17;
const double pinpitch = 1.26;
const double radi = 0.54;

const int ng = 7;
const int nXSset = 9;

const array<bool, 10> typeXS = { true,true,true,false,true,true,false,false,false,false };

const vector<vector<vector<double>>> XSSet = {
	{
		{	1.77949E-01, 3.29805E-01,	4.80388E-01, 5.54367E-01,	3.11801E-01, 3.95168E-01, 5.64406E-01 },
		{	8.02480E-03, 3.71740E-03,	2.67690E-02, 9.62360E-02,	3.00200E-02, 1.11260E-01, 2.82780E-01 },
		{	2.00600E-02, 2.02730E-03,	1.57060E-02, 4.51830E-02,	4.33421E-02, 2.02090E-01, 5.25711E-01 },
		{	2.00600E-02, 2.02730E-03,	1.57060E-02, 4.51830E-02,	4.33421E-02, 2.02090E-01, 5.25711E-01 },
		{	5.87910E-01, 4.11760E-01,	3.39060E-04, 1.17610E-07,	0.00000E+00, 0.00000E+00, 0.00000E+00 }
	},
	{
		{1.78731E-01,	3.30849E-01,	4.83772E-01,	5.66922E-01,	4.26227E-01,	6.78997E-01,	6.82852E-01},
		{8.43390E-03,	3.75770E-03,	2.79700E-02,	1.04210E-01,	1.39940E-01,	4.09180E-01,	4.09350E-01},
		{2.17530E-02,	2.53510E-03,	1.62680E-02,	6.54741E-02,	3.07241E-02,	6.66651E-01,	7.13990E-01},
		{2.17530E-02,	2.53510E-03,	1.62680E-02,	6.54741E-02,	3.07241E-02,	6.66651E-01,	7.13990E-01},
		{5.87910E-01,	4.11760E-01,	3.39060E-04,	1.17610E-07,	0.00000E+00,	0.00000E+00,	0.00000E+00}
	},
	{
		{1.81323E-01,	3.34368E-01,	4.93785E-01,	5.91216E-01,	4.74198E-01,	8.33601E-01,	8.53603E-01},
		{9.06570E-03,	4.29670E-03,	3.28810E-02,	1.22030E-01,	1.82980E-01,	5.68460E-01,	5.85210E-01},
		{2.38140E-02,	3.85869E-03,	2.41340E-02,	9.43662E-02,	4.57699E-02,	9.28181E-01,	1.04320E+00},
		{2.38140E-02,	3.85869E-03,	2.41340E-02,	9.43662E-02,	4.57699E-02,	9.28181E-01,	1.04320E+00},
		{5.87910E-01,	4.11760E-01,	3.39060E-04,	1.17610E-07,	0.00000E+00,	0.00000E+00,	0.00000E+00}

	},
	{
		{1.83045E-01,	3.36705E-01,	5.00507E-01,	6.06174E-01,	5.02754E-01,	9.21028E-01,	9.55231E-01},
		{9.48620E-03,	4.65560E-03,	3.62400E-02,	1.32720E-01,	2.08400E-01,	6.58700E-01,	6.90170E-01},
		{2.51860E-02,	4.73951E-03,	2.94781E-02,	1.12250E-01,	5.53030E-02,	1.07500E+00,	1.23930E+00},
		{2.51860E-02,	4.73951E-03,	2.94781E-02,	1.12250E-01,	5.53030E-02,	1.07500E+00,	1.23930E+00},
		{5.87910E-01,	4.11760E-01,	3.39060E-04,	1.17610E-07,	0.00000E+00,	0.00000E+00,	0.00000E+00}

	},
	{
		{1.26032E-01,	2.93160E-01,	2.84250E-01,	2.81020E-01,	3.34460E-01,	5.65640E-01,	1.17214E+00},
		{5.11320E-04,	7.58130E-05,	3.16430E-04,	1.16750E-03,	3.39770E-03,	9.18860E-03,	2.32440E-02},
		{1.32340E-08,	1.43450E-08,	1.12860E-06,	1.27630E-05,	3.53850E-07,	1.74010E-06,	5.06330E-06},
		{1.32340E-08,	1.43450E-08,	1.12860E-06,	1.27630E-05,	3.53850E-07,	1.74010E-06,	5.06330E-06},
		{5.87910E-01,	4.11760E-01,	3.39060E-04,	1.17610E-07,	0.00000E+00,	0.00000E+00,	0.00000E+00}
	},
	{
		{1.26032E-01,	2.93160E-01,	2.84240E-01,	2.80960E-01,	3.34440E-01,	5.65640E-01,	1.17215E+00},
		{5.11320E-04,	7.58010E-05,	3.15720E-04,	1.15820E-03,	3.39750E-03,	9.18780E-03,	2.32420E-02},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00}
	},
	{
		{1.59206E-01,	4.12970E-01,	5.90310E-01,	5.84350E-01,	7.18000E-01,	1.25445E+00,	2.65038E+00},
		{6.01050E-04,	1.57930E-05,	3.37160E-04,	1.94060E-03,	5.74160E-03,	1.50010E-02,	3.72390E-02},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00}
	},
	{
		{2.16768E-01,	4.80098E-01,	8.86369E-01,	9.70009E-01,	9.10482E-01,	1.13775E+00,	1.84048E+00},
		{1.70490E-03,	8.36224E-03,	8.37901E-02,	3.97797E-01,	6.98763E-01,	9.29508E-01,	1.17836E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00}
	},
	{
		{1.59206E-04,	4.12970E-04,	5.90310E-04,	5.84350E-04,	7.18000E-04,	1.25445E-03,	2.65038E-03},
		{6.01050E-07,	1.57930E-08,	3.37160E-07,	1.94060E-06,	5.74160E-06,	1.50010E-05,	3.72390E-05},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00},
		{0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00,	0.00000E+00}
	},
};

const vector<vector<double>> XSSM = {
	{
		{1.275370E-01,	4.237800E-02,	9.437400E-06,	5.516300E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	3.244560E-01,	1.631400E-03,	3.142700E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	4.509400E-01,	2.679200E-03,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	4.525650E-01,	5.566400E-03,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	1.252500E-04,	2.714010E-01,	1.025500E-02,	1.002100E-08,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	1.296800E-03,	2.658020E-01,	1.680900E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	8.545800E-03,	2.730800E-01}
	},
	{
		{1.288760E-01,	4.141300E-02,	8.229000E-06,	5.040500E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	3.254520E-01,	1.639500E-03,	1.598200E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	4.531880E-01,	2.614200E-03,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	4.571730E-01,	5.539400E-03,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	1.604600E-04,	2.768140E-01,	9.312700E-03,	9.165600E-09,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	2.005100E-03,	2.529620E-01,	1.485000E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	8.494800E-03,	2.650070E-01}
	},
	{
		{1.304570E-01,	4.179200E-02,	8.510500E-06,	5.132900E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	3.284280E-01,	1.643600E-03,	2.201700E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	4.583710E-01,	2.533100E-03,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	4.637090E-01,	5.476600E-03,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	1.761900E-04,	2.823130E-01,	8.728900E-03,	9.001600E-09,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	2.276000E-03,	2.497510E-01,	1.311400E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	8.864500E-03,	2.595290E-01}
	},
	{
		{1.315040E-01,	4.204600E-02,	8.697200E-06,	5.193800E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	3.304030E-01,	1.646300E-03,	2.600600E-09,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	4.617920E-01,	2.474900E-03,	0.000000E+00,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	4.680210E-01,	5.433000E-03,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	1.859700E-04,	2.857710E-01,	8.397300E-03,	8.928000E-09,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	2.391600E-03,	2.476140E-01,	1.232200E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	8.968100E-03,	2.560930E-01}
	},
	{
		{6.616590E-02,	5.907000E-02,	2.833400E-04,	1.462200E-06,	2.064200E-08,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	2.403770E-01,	5.243500E-02,	2.499000E-04,	1.923900E-05,	2.987500E-06,	4.214000E-07,
		0.000000E+00,	0.000000E+00,	1.834250E-01,	9.228800E-02,	6.936500E-03,	1.079000E-03,	2.054300E-04,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	7.907690E-02,	1.699900E-01,	2.586000E-02,	4.925600E-03,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	3.734000E-05,	9.975700E-02,	2.067900E-01,	2.447800E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	9.174200E-04,	3.167740E-01,	2.387600E-01,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	4.979300E-02,	1.099100E+00}
	},
	{
		{6.616590E-02,	5.907000E-02,	2.833400E-04,	1.462200E-06,	2.064200E-08,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	2.403770E-01,	5.243500E-02,	2.499000E-04,	1.923900E-05,	2.987500E-06,	4.214000E-07,
		0.000000E+00,	0.000000E+00,	1.832970E-01,	9.239700E-02,	6.944600E-03,	1.080300E-03,	2.056700E-04,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	7.885110E-02,	1.701400E-01,	2.588100E-02,	4.929700E-03,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	3.733300E-05,	9.973720E-02,	2.067900E-01,	2.447800E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	9.172600E-04,	3.167650E-01,	2.387700E-01,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	4.979200E-02,	1.099120E+00}
	},
	{
		{4.447770E-02,	1.134000E-01,	7.234700E-04,	3.749900E-06,	5.318400E-08,	0.000000E+00,	0.000000E+00,
		0.000000E+00,	2.823340E-01,	1.299400E-01,	6.234000E-04,	4.800200E-05,	7.448600E-06,	1.045500E-06,
		0.000000E+00,	0.000000E+00,	3.452560E-01,	2.245700E-01,	1.699900E-02,	2.644300E-03,	5.034400E-04,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	9.102840E-02,	4.155100E-01,	6.373200E-02,	1.213900E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	7.143700E-05,	1.391380E-01,	5.118200E-01,	6.122900E-02,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	2.215700E-03,	6.999130E-01,	5.373200E-01,
		0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	0.000000E+00,	1.324400E-01,	2.480700E+00}
	},
	{
		{1.705630E-01, 	4.440120E-02, 9.836700E-05, 1.277860E-07, 0.000000E+00, 0.000000E+00,	0.000000E+00,
		0.000000E+00, 	4.710500E-01, 6.854800E-04, 3.913950E-10, 0.000000E+00, 0.000000E+00,	0.000000E+00,
		0.000000E+00, 	0.000000E+00, 8.018590E-01, 7.201320E-04, 0.000000E+00, 0.000000E+00,	0.000000E+00,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 5.707520E-01, 1.460150E-03, 0.000000E+00,	0.000000E+00,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 6.555620E-05, 2.078380E-01, 3.814860E-03,	3.697600E-09,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 0.000000E+00, 1.024270E-03, 2.024650E-01,	4.752900E-03,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 3.530430E-03,	6.585970E-01}
	},
	{
		{4.447770E-05, 	1.134000E-04, 7.234700E-07, 3.749900E-09, 5.318400E-11, 0.000000E+00, 0.000000E+00,
		0.000000E+00, 	2.823340E-04, 1.299400E-04, 6.234000E-07, 4.800200E-08, 7.448600E-09, 1.045500E-09,
		0.000000E+00, 	0.000000E+00, 3.452560E-04, 2.245700E-04, 1.699900E-05, 2.644300E-06, 5.034400E-07,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 9.102840E-05, 4.155100E-04, 6.373200E-05, 1.213900E-05,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 7.143700E-08, 1.391380E-04, 5.118200E-04, 6.122900E-05,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 0.000000E+00, 2.215700E-06, 6.999130E-04, 5.373200E-04,
		0.000000E+00, 	0.000000E+00, 0.000000E+00, 0.000000E+00, 0.000000E+00, 1.324400E-04, 2.480700E-03}
	}
};