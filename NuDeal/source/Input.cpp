#include "Input.h"
#include "IOUtil.h"
#include "Exception.h"
#include "IOUtil.h"
#include "Preprocessor.h"
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

		auto& T = children.emplace_back();
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
	stringstream stream;

	stream << fin.rdbuf();
	
	contents = stream.str();

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
		
		UnitVolume_t& U = unitVolumes[name];

		for (auto& s : v) {
			if (Util::Uppercase(s).find(ORIGIN) != string::npos) {
				auto b = s.find_first_of(SC::LeftParen);
				auto e = s.find_last_of(SC::RightParen);
				auto origin = Util::SplitFields(s.substr(b + 1, e - b - 1), SC::Comma);
				if (origin.size() != 3) Except::Abort(Code::INVALID_COORDINATE, s);
				U.origin.x = Util::Float(origin[0]);
				U.origin.y = Util::Float(origin[1]);
				U.origin.z = Util::Float(origin[2]);
			}
			else U.equations.emplace_back(Parser_t::ParseEquation(s));
		}
	}

}

void InputManager_t::ParseUnitCompCard(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	static const string disp = string(2, SC::RightAngle);
	static const string ORIGIN = "ORIGIN";

	for (auto& T : Tree.children) {
		auto prefix = Util::EraseSpace(T.name);
		auto v = Util::SplitFields(prefix, SC::Colon);
		if (v.size() != 2)
			Except::Abort(Code::BACKGROUND_MISSED, T.contents, T.GetLineInfo());
		auto name = v.front();
		auto background = v.back();
		auto contents = Util::EraseSpace(T.contents);
		v = Util::SplitFields(contents, SC::SemiColon);

		UnitComp_t& U = unitComps[name];
		U.background = background;

		for (const auto& s : v) {
			if (Util::Uppercase(s.substr(0, ORIGIN.size())) == ORIGIN) {
				auto b = s.find_first_of(SC::LeftParen);
				auto e = s.find_last_of(SC::RightParen);
				auto origin = Util::SplitFields(s.substr(b + 1, e - b - 1), SC::Comma);
				if (origin.size() != 3) Except::Abort(Code::INVALID_COORDINATE, s);
				U.origin.x = Util::Float(origin[0]);
				U.origin.y = Util::Float(origin[1]);
				U.origin.z = Util::Float(origin[2]);
			}
			else {
				auto u = Util::SplitFields(s, disp);
				U.unitvols.emplace_back(u.front());
				auto& D = U.displace.emplace_back();
				for (auto k = u.begin() + 1; k < u.end(); ++k) {
					auto& d = D.emplace_back();
					auto& s = *k;
					if (std::find_if(s.begin(), s.end(), isalpha) != s.end()) { // Rotation
						d.type = Displace_t::Type::Rotation;
						auto r = Util::SplitFields(s, SC::Equal);
						if (r.size() != 2)
							Except::Abort(Code::INVALID_ROTATION, s);
						switch (toupper(r[0][0])) {
							case 'X' : d.axis = Displace_t::Axis::X; break;
							case 'Y' : d.axis = Displace_t::Axis::Y; break;
							case 'Z' : d.axis = Displace_t::Axis::Z; break;
							default : Except::Abort(Code::INVALID_ROTATION, s);
						}
						d.move[static_cast<int>(d.axis)] = Util::Float(r[1]);
					}
					else { // Translation
						d.type = Displace_t::Type::Translation;
						d.axis = Displace_t::Axis::INVALID;
						auto t = Util::EraseSpace(s, "()");
						auto v = Util::SplitFields(t, SC::Comma);
						if (v.size() != 3) Except::Abort(Code::INVALID_COORDINATE, s);
						for (int i = 0; i < 3; ++i)
							d.move[i] = Util::Float(v[i]);
					}
				}
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