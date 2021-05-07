#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{

class Parser_t
{
public :
	
	using Util = Util_t;
	using Except = Exception_t;
	using SC = SpecialCharacters;

	struct Equation_t
	{
		double c[10];
	};

	struct UnitVolume_t
	{
		double3 origin;
		vector<Equation_t> eqs;
	};

	Parser_t(const Lexer_t *Lexer) noexcept;

	void Parse();

private :
	
	int depth = 0;
	const vector<Token_t>& tokens;
	vector<Token_t>::const_iterator iter;

	// Blocks
	void ScanGeometryBlock();
	void ScanMaterialBlock();
	void ScanOptionBlock();

	// Cards
	void ScanUnitVolumeCard();
	UnitVolume_t GetUnitVolume();

};


}