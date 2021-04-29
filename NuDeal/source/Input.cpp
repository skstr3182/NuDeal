#include "Input.h"
#include "IOUtil.h"
#include "Exception.h"

namespace IO
{

InputManager_t::Blocks InputManager_t::GetBlockID(string line) const
{
	int pos_end = line.find(SC::BLANK, 0);
	string block = line.substr(0, pos_end);

	Uppercase(block);
	for (int i = 0; i < num_blocks; ++i)
		if (!block.compare(BlockNames[i]))
			return static_cast<Blocks>(i);
	return Blocks::INVALID;
}

template <typename T>
T InputManager_t::GetCardID(Blocks block, string line) const
{
	int pos_beg = line.find_first_not_of(SC::BLANK);
	if (pos_beg == string::npos) return static_cast<T>(INVALID);

	int pos_end = line.find(SC::BLANK, pos_beg);
	string card = line.substr(pos_beg, pos_end - pos_beg);

	Uppercase(card);
	int b = static_cast<int>(block);
	for (int i = 0; i < num_cards; ++i) 
		if (!card.compare(CardNames[b][i])) 
			return static_cast<T>(i);
	return static_cast<T>(INVALID);
}

stringstream InputManager_t::ExtractInput(istream& fin, string& contents) const
{
	stringstream fileStream;

	fileStream << fin.rdbuf();
	
	contents = fileStream.str();

	Uppercase(contents);

	string replaced = contents;

	std::replace(replaced.begin(), replaced.end(), SC::TAB, SC::BLANK);
	
	fileStream.str("");
	fileStream.str(replaced);

	return static_cast<stringstream&&>(fileStream);
}

// BLOCK

void InputManager_t::ParseGeometryBlock(istream& in)
{
	using Except = Exception_t;
	using Code = Except::Code;
	using Cards = GeometryCards;

	while (!in.eof()) {
		
		stringstream sector(GetScriptBlock(in));
		string block = GetLine(sector, SC::LBRACE);
		Cards ID = GetCardID<Cards>(Blocks::GEOMETRY, block);

		switch (ID)
		{
		case Cards::UNITVOLUME :
			ParseUnitVolumeCard(sector); break;
		case Cards::UNITCOMP :
			ParseUnitCompCard(sector); break;
		case Cards::INVALID :
			Except::Abort(Code::INVALID_INPUT_CARD, block);
		}

	}
}

void InputManager_t::ParseMaterialBlock(istream& in)
{

}

void InputManager_t::ParseOptionBlock(istream& in)
{

}

// GEOMETRY CARDS

void InputManager_t::ParseUnitVolumeCard(istream& in)
{
	using Except = Exception_t;
	using Code = Except::Code;

	while (!in.eof()) {
			
		stringstream sector(GetScriptBlock(in));
		if (sector.str().empty()) break;
		auto name = GetLine(sector, SC::LBRACE);
		auto info = GetLine(sector, SC::RBRACE);
		info.erase(std::remove(info.begin(), info.end(), SC::BLANK), info.end());

		UnitVolume_t V;

		string::size_type pos = info.find("ORIGIN");

		if (pos != string::npos) {
			auto end = info.find_first_of(SC::SEMICOLON, pos);
			if (end == string::npos) 
				Except::Abort(Code::SEMICOLON_MISSED, info);
			auto substr = info.substr(pos, end - pos);
			info.erase(pos, min(end + 1, info.size()));
			auto v = SplitFields(substr, &SC::COLON);
			V.origin = v.back();
		}

		auto v = SplitFields(info, SC::DAMPERSAND);
		V.equations = v;
		unitVolumes[name] = V;

	}

}

void InputManager_t::ParseUnitCompCard(istream& in)
{

}

// PUBLIC

void InputManager_t::ReadInput(string file)
{
	using Except = Exception_t;
	using Code = Except::Code;

	ifstream fin(file);
	
	if (fin.fail()) Except::Abort(Code::FILE_NOT_FOUND, file);

	stringstream fileStream = ExtractInput(fin, contents);

	fin.close();

	while (!fileStream.eof()) {
		
		stringstream sector(GetScriptBlock(fileStream));
		string block = GetLine(sector, SC::LBRACE);
		Blocks ID = GetBlockID(block);

		switch (ID)
		{
		case Blocks::GEOMETRY :
			ParseGeometryBlock(sector); break;
		case Blocks::MATERIAL :
			ParseMaterialBlock(sector); break;
		case Blocks::OPTION : 
			ParseOptionBlock(sector); break;
		case Blocks::INVALID :
			Except::Abort(Code::INVALID_INPUT_BLOCK, block);
		}
	}
}

}