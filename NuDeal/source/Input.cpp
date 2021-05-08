#include "Input.h"
#include "IOUtil.h"
#include "Exception.h"
#include "IOUtil.h"
#include "Preprocessor.h"
#include "Lexer.h"
#include "Parser.h"

namespace IO
{

void InputManager_t::HashTree_t::Make(const string& file, 
	string::size_type Beg, string::size_type End)
{
	string::size_type beg = Beg, end;

	while ((end = Util::FindEndPoint(file, beg)) < End) {
		auto name = file.substr(Beg, beg - Beg);

		beg = file.find_first_not_of(SC::LeftBrace, beg);
		auto contents = file.substr(beg, end - beg);

		children.emplace_back();
		auto& T = children.back();
		T.name = Util::Trim(name);
		T.parent = this;
		T.CountLine(name, contents);

		if (Util::IsClosed(contents)) 
			T.Make(file, beg, end);
		else 
			T.contents = contents;

		beg = end + 1; Beg = beg;
	}
}

void InputManager_t::HashTree_t::CountLine(const string& name, const string& contents)
{
	num_lines = Util::LineCount(name) + Util::LineCount(contents);
	line_info = Util::LineCount(name, name.find_first_not_of(SC::LF)) + 1;
	line_info += parent->line_info;
	for (const auto& t : parent->children) {
		if (&t != this) line_info += t.num_lines;
	}
}


void InputManager_t::ExtractInput(istream& fin)
{
	stringstream strstream;

	strstream << fin.rdbuf();
	
	contents = strstream.str();

	std::replace(contents.begin(), contents.end(), SC::Tab, SC::Blank);
	std::replace(contents.begin(), contents.end(), SC::CR, SC::Blank);
}

// BLOCK

void InputManager_t::ParseGeometryBlock(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;
	using Cards = GeometryCards;

	for (auto& T : Tree.children) {
		
		string card = Util::Trim(T.name);
		Cards ID = Util::GetCardID<Cards>(Blocks::GEOMETRY, card);

		switch (ID)
		{
		case Cards::UNITVOLUME :
			ParseUnitVolumeCard(T); break;
		case Cards::UNITCOMP :
			ParseUnitCompCard(T); break;
		case Cards::INVALID :
			Except::Abort(Code::INVALID_INPUT_CARD, card, T.GetLineInfo());
		}

	}

}

void InputManager_t::ParseMaterialBlock(HashTree_t& Tree)
{

}

void InputManager_t::ParseOptionBlock(HashTree_t& Tree)
{

}

/// GEOMETRY CARDS

void InputManager_t::ParseUnitVolumeCard(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	static const string ORIGIN = "ORIGIN";

	for (auto& T : Tree.children) {
		auto name = Util::Trim(T.name);
		auto contents = Util::EraseSpace(T.contents);
		auto v = Util::SplitFields(contents, SC::SemiColon);
		
		UnitVolume_t U;

		for (const auto& s : v) {
			if (s.find(ORIGIN) != string::npos) {
				auto b = s.find_first_of(SC::LeftParen);
				auto e = s.find_last_of(SC::RightParen);
				auto origin = Util::SplitFields(s.substr(b + 1, e - b - 1), {SC::Comma});
				if (origin.size() != 3) Except::Abort(Code::INVALID_ORIGIN_DATA, s);
				U.origin.x = Util::Float(origin[0]);
				U.origin.y = Util::Float(origin[1]);
				U.origin.z = Util::Float(origin[2]);
			}
			else {

			}
		}
		v = Util::SplitFields(v.front(), SC::SemiColon);

		U.equations = v;

		unitVolumes[name] = std::move(U);
	}

}

void InputManager_t::ParseUnitCompCard(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	static const string disp = string(2, SC::RightAngle);

	for (auto& T : Tree.children) {
		auto prefix = Util::EraseSpace(T.name);
		if (std::count(prefix.begin(), prefix.end(), SC::Colon) != 1)
			Except::Abort(Code::BACKGROUND_MISSED, T.contents, T.GetLineInfo());
		auto v = Util::SplitFields(prefix, SC::Colon);
		auto name = v.front();
		auto background = v.back();
		auto contents = Util::EraseSpace(T.contents);
		v = Util::SplitFields(contents, SC::SemiColon);

		UnitComp_t U;

		U.background = background;

		for (const auto& s : v) {
			auto u = Util::SplitFields(s, disp);
			U.unitvols.push_back(u.front());
			U.displace.push_back(vector<string>());
			u.erase(u.begin());
			for (const auto& k : u) {
				auto lpos = k.find(SC::LeftParen) + 1;
				auto rpos = k.find(SC::RightParen);
				if (lpos == string::npos && rpos == string::npos)
					U.displace.back().push_back(k);
				else if (lpos != string::npos && rpos != string::npos)
					U.displace.back().push_back(k.substr(lpos, rpos - lpos));
				else
					Except::Abort(Code::MISMATCHED_BRAKETS, T.contents, T.GetLineInfo());
			}
		}

		unitComps[name] = std::move(U);
	}
}

void InputManager_t::Preprocess()
{
	using Preprocessor = Preprocessor_t;
	using Except = Exception_t;
	using Code = Except::Code;

	try {
		Preprocessor::DeleteComment(contents);
	}
	catch (exception& e) {
		Except::Abort(Code::MISMATCHED_BRAKETS, e.what());
	}

	try {
		Preprocessor::CheckBalance(contents, {"{", "[", "("}, {"}", "]", ")"} );
	}
	catch (exception& e) {
		Except::Abort(Code::MISMATCHED_BRAKETS, e.what());
	}

	Preprocessor::RemoveBlankInParenthesis(contents);

	try {
		Preprocessor::ApplyMacro(contents);
	}
	catch (exception& e) {
		Except::Abort(Code::INVALID_MACRO, e.what());
	}

	contents.erase(std::remove(contents.begin(), contents.end(), SC::Blank), contents.end());

	Lexer_t Lexer;

	Lexer.Lex(contents);
}

void InputManager_t::ReadInput(string file)
{
	using Except = Exception_t;
	using Code = Except::Code;

	ifstream fin(file);
	
	if (fin.fail()) Except::Abort(Code::FILE_NOT_FOUND, file);

	this->file = file;

	ExtractInput(fin);

	fin.close();

	Preprocess();

	HashTree.Make(contents);
	
	for (auto& T : HashTree.children) {
		string block = Util::Trim(T.name);
		Blocks ID = Util::GetBlockID(block);
		switch (ID)
		{
		case Blocks::GEOMETRY :
			ParseGeometryBlock(T); break;
		case Blocks::MATERIAL :
			ParseMaterialBlock(T); break;
		case Blocks::OPTION :
			ParseOptionBlock(T); break;
		case Blocks::INVALID :
			Except::Abort(Code::INVALID_INPUT_BLOCK, block, T.GetLineInfo());
		}
	}


}

}