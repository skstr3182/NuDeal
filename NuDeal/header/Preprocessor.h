#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{

class Preprocessor_t
{
public :

	using size_type = string::size_type;
	using SC = SpecialCharacters;
	using Util = Util_t;

private :

	static const string macro_str;
	static const regex macro_re;
	
public :
	
	static void DeleteComment(string& contents);
	static void ApplyMacro(string& contents);
	static void CheckBalance(const string& contents,
		const vector<string>& open, const vector<string>& close);


};

}