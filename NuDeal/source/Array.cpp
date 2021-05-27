#include "Array.hpp"

namespace LinPack
{

template <typename T>
void Array_t<T>::Fill(const T& value)
{
	//if (d_state != State::Undefined) FillDevice(value);
	//if (state != State::Undefined) *this = value;
}

template class Array_t<int>;
template class Array_t<bool>;
template class Array_t<float>;
template class Array_t<double>;

}