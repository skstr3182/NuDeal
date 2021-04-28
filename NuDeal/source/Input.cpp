#include "Input.h"
#include "IOUtil.h"

namespace IO
{

InputManager_t::Blocks InputManager_t::GetBlockID(string line) const
{
	int pos_end = line.find(BLANK, 0);
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
	int pos_beg = line.find_first_not_of(BLANK);
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

}