#include "XSLibrary.h"
#include "IOUtil.h"
#include "IOExcept.h"

namespace Library
{


void XSLibrary_t::ReadMacro(const string& file)
{
	using IOUtil = IO::Util_t;
	using Except = IO::Exception_t;

	ifstream fin(file);

	if (fin.fail()) Except::Abort(Except::Code::FILE_NOT_FOUND);
	


}

}