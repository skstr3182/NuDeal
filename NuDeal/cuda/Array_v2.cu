#include "Array_v2.h"

namespace LinPack_v2
{

template <typename T>
bool Array_t<T, is_device_t<T>>::_check_dims(
	size_type nx,
	size_type ny,
	size_type nz,
	size_type nw
) const
{
	if (this->nx != nx) return false;
	if (this->ny != ny) return false;
	if (this->nz != nz) return false;
	if (this->nw != nw) return false;
	return true;
}

template <typename T>
void Array_t<T, is_device_t<T>>::ResizeDevice()
{
	container_device.resize(n);
	ptr_device = thrust::raw_pointer_cast(container_device.data());
}

template <typename T>
void Array_t<T, is_device_t<T>>::ResizeDevice(
	size_type nx,
	size_type ny,
	size_type nz,
	size_type nw
)
{
	if (IsAlloc() && !_check_dims(nx, ny, nz, nw)) {
		// Except
	}
	else {
		SetDimension(nx, ny, nz, nw);
		ResizeDevice();
	}
}

template class Array_t<double, is_device_t<double>>;
}