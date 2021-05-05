#pragma once
#include "Defines.h"
#include "IODeclare.h"

namespace IO
{

class Util_t
{
public :

	using SC = SpecialCharacters;
	using size_type = string::size_type;

	// String Parsing Utility
	/// Type Conversion
	static string Uppercase(string& line);
	static int Integer(string field);
	static double Float(string field);
	static bool Logical(string field);
	/// String Manipulation
	static string Trim(const string& field, const string& delimiter = "\n ");
	static int LineCount(const string& line, size_type count = string::npos);
	static string EraseSpace(const string& line, const string& delimiter = "\n ");
	/// Macro Syntax Manipulation
	static size_type FindEndOfMacro(const string& line, size_type pos = 0);
	static string ReplaceMacro(const string& line, char c = SC::BLANK);
	static vector<string> ExtractMacro(const string& line);
	/// Read Input
	static string GetLine(stringstream& in, const char delimiter = SC::LF);
	static string GetBlock(stringstream& fin);
	static size_type FindEndPoint(const string& contents, size_type& pos);
	static string GetBlock(const string& contents, size_type pos);
	static void ReplaceComments(string& line);
	static vector<string> SplitFields(string line, const string& delimiter);
	
	// Check Synax Error
	static void AreBracketsMatched(const string& contents);
	static void IsMacroValid(const string& contents);
	static void IsVariableCorrect(const string& contents);
	static bool IsClosed(const string& s);
};



}