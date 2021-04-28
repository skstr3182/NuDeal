#pragma once
#include <string>

using namespace std;

const char BLANK = ' ';
const char BANG = '!';
const char BRACKET = '>';
const char TAB = '\t';
const char CR = '\r';
const char LBRACE = '{';
const char RBRACE = '}';

const int num_blocks = 2;
const string BlockName[num_blocks]{
	"GEOM",
	"MacroXS"
};

enum class Blocks {
	GEOM,
	MacroXS
};

const int num_cards = 30;
const string CardName[num_blocks][num_cards]{
	// GEOM
	{"UnitVolume","UnitComp","Displace"},
	{"NG","FORMAT"}
};

int procinp(char *Filename);