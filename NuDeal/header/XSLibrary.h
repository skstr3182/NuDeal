#pragma once
#include "Defines.h"
#include "Array.h"

namespace Library
{

class XSLibrary_t
{
public:

	template <typename T>
	using Array = LinPack::Array_t<T>;

	enum class XSType
	{
		Micro,
		Macro
	};

	enum class MicroType
	{
		TOT,
		ABS,
		FIS,
		SCAT,
		NU,
		CHI,
		KAPPA,
		CAP,
		N2N,
		N3N,
		COUNT
	};

	enum class MacroType
	{
		TOT,
		ABS,
		NUFIS,
		KAPPAFIS,
		CHI,
		SCAT,
		COUNT
	};

	template <typename T>
	struct Data_t
	{
		Array<double> xs[static_cast<int>(T::COUNT)];
		Array<double> scat_matrix[3];
	};

private:

	int type = static_cast<int>(XSType::Micro);
	int num_isotopes = 0, num_groups = 0;
	int num_temps = 0, scattering_order = 0;

	std::vector<Data_t<MacroType>> macroData;
	std::vector<Data_t<MicroType>> microData;

public:

	XSLibrary_t() noexcept = default;

	void ReadMacro(const string& file = "./lib/c5g7.xslib");
	void ReadMicro();

public:

	bool IsMacro() const { return !macroData.empty(); }
	bool IsMicro() const { return !microData.empty(); }
};

}