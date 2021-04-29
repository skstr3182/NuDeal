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

	int pos_end = line.find(BLANK, pos_beg);
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

	string replaced = contents;

	std::replace(replaced.begin(), replaced.end(), SC::TAB, SC::BLANK);
	
	fileStream.str("");
	fileStream.str(replaced);

	return static_cast<stringstream&&>(fileStream);
}

void InputManager_t::ReadInput(string file)
{
	using Except = Exception_t;
	using Code = Except::Code;

	ifstream fin(file);
	
	if (fin.fail()) Except::Abort(Code::FILE_NOT_FOUND, file);

	stringstream fileStream = ExtractInput(fin, contents);

	fin.close();

	cout << GetLine(fileStream, SC::LBRACE) << endl;
	cout << GetLine(fileStream, SC::LBRACE) << endl;
	fileStream.clear(stringstream::goodbit); fileStream.seekg(0);
	cout << GetLine(fileStream) << endl;
	cout << GetLine(fileStream) << endl;

	cout << CountCurrentLine(fileStream) << endl;
}

}