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

	for (auto& T : Tree.children) {
		auto fields = Util::SplitFields(T.name, "()");
		string name, origin;

		if (fields.size() == 1)
			name = Util::Trim(fields.front());
		else if (fields.size() == 2) {
			name = Util::Trim(fields.front());
			origin = Util::EraseSpace(fields.back());
		}
		else
			Except::Abort(Code::INVALID_UNIT_GEOMETRY, T.name, T.GetLineInfo());

		if (name.find(SC::Blank) != string::npos)
			Except::Abort(Code::INVALID_UNIT_GEOMETRY, T.name, T.GetLineInfo());
		
		auto& U = unitVolumes[name];
		if (!origin.empty()) U.origin = Util::GetCoordinate(origin);

		auto contents = Util::EraseSpace(T.contents);
		fields = Util::SplitFields(contents, SC::SemiColon);
		for (auto& s : fields) U.equations.emplace_back(Parser_t::ParseEquation(s));

	}

}

void InputManager_t::ParseUnitCompCard(HashTree_t& Tree)
{
	using Except = Exception_t;
	using Code = Except::Code;

	static const string disp = string(2, SC::RightAngle);

	for (auto& T : Tree.children) {
		auto fields = Util::SplitFields(T.name, "()");
		string name, origin;

		if (fields.size() == 1) name = Util::Trim(fields.front());
		else if (fields.size() == 2) {
			name = Util::Trim(fields.front());
			origin = Util::EraseSpace(fields.back());
		}
		else
			Except::Abort(Code::INVALID_UNIT_GEOMETRY, T.name, T.GetLineInfo());

		if (name.find(SC::Blank) != string::npos)
			Except::Abort(Code::INVALID_UNIT_GEOMETRY, T.name, T.GetLineInfo());

		auto& U = unitComps[name];
		if (!origin.empty()) U.origin = Util::GetCoordinate(origin);

		auto contents = Util::EraseSpace(T.contents);
		fields = Util::SplitFields(contents, SC::SemiColon);

		for (const auto& s : fields) {
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
					if (r[0] == "X") 
						d.axis = Displace_t::Axis::X;
					else if (r[0] == "Y") 
						d.axis = Displace_t::Axis::Y;
					else if (r[0] == "Z") 
						d.axis = Displace_t::Axis::Z;
					else 
						Except::Abort(Code::INVALID_ROTATION, s);
					d.move[static_cast<int>(d.axis)] = Util::Float(r[1]);
				}
				else { // Translation
					d.type = Displace_t::Type::Translation;
					d.axis = Displace_t::Axis::INVALID;
					auto coordinate = Util::GetCoordinate(s);
					d.move[0] = coordinate.x;
					d.move[1] = coordinate.y;
					d.move[2] = coordinate.z;
				}
			}
		}
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
	catch (const exception& e) {
		Except::Abort(Code::MISMATCHED_BRAKETS, e.what());
	}

	try {
		Preprocessor::CheckBalance(contents, {"{", "[", "("}, {"}", "]", ")"} );
	}
	catch (const exception& e) {
		Except::Abort(Code::MISMATCHED_BRAKETS, e.what());
	}

	Preprocessor::RemoveBlankInParenthesis(contents);

	try {
		Preprocessor::ApplyMacro(contents);
	}
	catch (const exception& e) {
		Except::Abort(Code::INVALID_MACRO, e.what());
	}

	contents.erase(std::remove(contents.begin(), contents.end(), SC::Blank), contents.end());
}

void InputManager_t::ReadInput(const string& file)
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