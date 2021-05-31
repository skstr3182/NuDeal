#pragma once
#include "mpi.h"
#include "cuda_runtime.h"
#include <type_traits>

namespace MPI
{

namespace // Anonymous Namespace
{
template <typename T, typename = void> class Datatype;

// Built-in Datatype
template <typename T>
class Datatype<T, typename std::enable_if<std::is_scalar<T>::value>::type>
{
public:
	static const MPI_Datatype type;
};

// User-defined Datatype
template <typename T>
class Datatype<T, typename std::enable_if<!std::is_scalar<T>::value>::type>
{
public:
	inline static MPI_Datatype type = MPI_DATATYPE_NULL;
	inline static int Commit(MPI_Datatype *t)
	{
		type = *t;
		return MPI_Type_commit(&type);
	}
};


// Built-in Datatype

const MPI_Datatype Datatype<bool>::type						= MPI_C_BOOL;
const MPI_Datatype Datatype<char>::type						= MPI_CHAR;
const MPI_Datatype Datatype<unsigned char>::type	= MPI_UNSIGNED_CHAR;
const MPI_Datatype Datatype<short>::type					= MPI_SHORT;
const MPI_Datatype Datatype<unsigned short>::type	= MPI_UNSIGNED_SHORT;
const MPI_Datatype Datatype<int>::type						= MPI_INT;
const MPI_Datatype Datatype<unsigned int>::type		= MPI_UNSIGNED;
const MPI_Datatype Datatype<long>::type						= MPI_LONG;
const MPI_Datatype Datatype<unsigned long>::type  = MPI_UNSIGNED_LONG;
const MPI_Datatype Datatype<long long>::type			= MPI_LONG_LONG;
const MPI_Datatype Datatype<float>::type					= MPI_FLOAT;
const MPI_Datatype Datatype<double>::type					= MPI_DOUBLE;
const MPI_Datatype Datatype<long double>::type		= MPI_LONG_DOUBLE;

// Alias

template <typename _Ty>
inline const MPI_Datatype type_v = Datatype<_Ty>::type;

} // Anonymous Namespace

// Basic MPI Function Template Inliner

/*---------------------------------------------*/
/* Section 3.2: Blocking Communication         */
/*---------------------------------------------*/

template <typename T> 
inline constexpr int Send(const T *buf, 
	int count, 
	int dest, 
	int tag, 
	MPI_Comm comm)
{
	return MPI_Send(buf, count, type_v<T>, dest, tag, comm);
}

template <typename T>
inline constexpr int Recv(T *buf, 
	int count, 
	int source, 
	int tag, 
	MPI_Comm comm, 
	MPI_Status *stat = MPI_STATUS_IGNORE)
{
	return MPI_Recv(buf, count, type_v<T>, source, tag, comm, stat);
}

/*---------------------------------------------*/
/* Section 3.7: Nonblocking Communication      */
/*---------------------------------------------*/

template <typename T>
inline constexpr int Isend(const T *buf, 
	int count, 
	int dest, 
	int tag, 
	MPI_Comm comm, 
	MPI_Request *request)
{
	return MPI_Isend(buf, count, type_v<T>, dest, tag, comm, request);
}

template <typename T>
inline constexpr int Irecv(T *buf,
	int count,
	int source,
	int tag,
	MPI_Comm comm,
	MPI_Request *request)
{
	return MPI_Irecv(buf, count, type_v<T>, source, tag, comm, request);
}

/*---------------------------------------------*/
/* Section 3.7.3: Communication Completion     */
/*---------------------------------------------*/

inline constexpr int (*Wait)(MPI_Request*, MPI_Status*) = MPI_Wait;

/*---------------------------------------------*/
/* Section 3.7.5: Multiple Completions         */
/*---------------------------------------------*/

inline constexpr int (*Waitall)(int, MPI_Request[], MPI_Status[]) = MPI_Waitall;

/*---------------------------------------------*/
/* Section 4.1: Derived Datatypes              */
/*---------------------------------------------*/

inline constexpr int (*Type_contiguous)(int, MPI_Datatype, MPI_Datatype*) = MPI_Type_contiguous;

/*---------------------------------------------*/
/* Section 4.1.9: Datatype Commit and Free     */
/*---------------------------------------------*/

template <typename T>
inline constexpr int Type_commit(MPI_Datatype *datatype)
{
	return Datatype<T>::Commit(datatype);
}

inline constexpr int (*Type_free)(MPI_Datatype*) = MPI_Type_free;

/*---------------------------------------------*/
/* Section 5.3: Barrier Synchronization        */
/*---------------------------------------------*/

inline constexpr int (*Barrier)(MPI_Comm) = MPI_Barrier;

/*---------------------------------------------*/
/* Section 5.4: Broadcast                      */
/*---------------------------------------------*/

template <typename T>
inline constexpr int Bcast(T *buffer,
	int count,
	int root,
	MPI_Comm comm)
{
	return MPI_Bcast(buffer, count, type_v<T>, root, comm);
}

/*---------------------------------------------*/
/* Section 5.5: Gather                         */
/*---------------------------------------------*/

template <typename T, typename U>
inline constexpr int Gather(const T *sendbuf,
	int sendcount,
	U *recvbuf,
	int recvcount,
	int root,
	MPI_Comm comm)
{
	return MPI_Gather(sendbuf, sendcount, type_v<T>,
		recvbuf, recvcount, type_v<U>, root, comm);
}

template <typename T, typename U>
inline constexpr int Gatherv(const T *sendbuf,
	int sendcount,
	U *recvbuf,
	const int recvcounts[],
	const int displs[],
	int root,
	MPI_Comm comm)
{
	return MPI_Gatherv(sendbuf, sendcount, type_v<T>, 
		recvbuf, recvcounts, displs, type_v<U>, root, comm);
}

/*---------------------------------------------*/
/* Section 5.6: Scatter                        */
/*---------------------------------------------*/

template <typename T, typename U>
inline constexpr int Scatter(const T *sendbuf,
	int sendcount,
	U *recvbuf,
	int recvcount,
	int root,
	MPI_Comm comm)
{
	return MPI_Scatter(sendbuf, sendcount, type_v<T>,
		recvbuf, recvcount, type_v<U>, root, comm);
}

template <typename T, typename U>
inline constexpr int Scatterv(const T *sendbuf,
	const int sendcounts[],
	const int displs[],
	U *recvbuf,
	int recvcount,
	int root,
	MPI_Comm comm)
{
	return MPI_Scatterv(sendbuf, sendcounts, displs, type_v<T>,
		recvbuf, recvcount, type_v<U>, root, comm);
}

/*---------------------------------------------*/
/* Section 5.6: Gather-to-all                  */
/*---------------------------------------------*/

template <typename T, typename U>
inline constexpr int Allgather(const T *sendbuf,
	int sendcount,
	U *recvbuf,
	int recvcount,
	MPI_Comm comm)
{
	return MPI_Allgather(sendbuf, sendcount, type_v<T>,
		recvbuf, recvcount, type_v<U>, comm);
}

template <typename T, typename U>
inline constexpr int Allgatherv(const T *sendbuf,
	int sendcount,
	U *recvbuf,
	const int recvcounts[],
	const int displs[],
	MPI_Comm comm)
{
	return MPI_Allgatherv(sendbuf, sendcount, type_v<T>,
		recvbuf, recvcounts, displs, type_v<U>, comm);
}

/*---------------------------------------------*/
/* Section 5.9: Global Reduction Operations    */
/*---------------------------------------------*/

template <typename T>
inline constexpr int Reduce(const T* sendbuf, 
	T *recvbuf, 
	int count, 
	MPI_Op op, 
	int root, 
	MPI_Comm comm)
{
	return MPI_Reduce(sendbuf, recvbuf, count, type_v<T>, op, root, comm);
}

template <typename T>
inline constexpr int Allreduce(const T* sendbuf, 
	T *recvbuf, 
	int count, 
	MPI_Op op, 
	MPI_Comm comm)
{
	return MPI_Allreduce(sendbuf, recvbuf, count, type_v<T>, op, comm);
}

/*---------------------------------------------*/
/* Section 6.4: Communicator Management        */
/*---------------------------------------------*/

inline constexpr int (*Comm_size)(MPI_Comm, int*) = MPI_Comm_size;
inline constexpr int (*Comm_rank)(MPI_Comm, int*) = MPI_Comm_rank;
inline constexpr int (*Comm_create)(MPI_Comm, MPI_Group, MPI_Comm*) = MPI_Comm_create;
inline constexpr int (*Comm_split)(MPI_Comm, int, int, MPI_Comm*) = MPI_Comm_split;
inline constexpr int (*Comm_split_type)(MPI_Comm, int, int, MPI_Info, MPI_Comm*) = MPI_Comm_split_type;
inline constexpr int (*Comm_free)(MPI_Comm*) = MPI_Comm_free;

/*---------------------------------------------*/
/* Section 8.7: Startup                        */
/*---------------------------------------------*/

inline constexpr int (*Finalize)() = MPI_Finalize;
inline constexpr int (*Abort)(MPI_Comm, int) = MPI_Abort;
inline constexpr int (*Init)(const int* , char***) = MPI_Init;

inline constexpr auto Configure_cuda_types = []()
{
	MPI_Datatype MPI_INT2, MPI_INT3, MPI_INT4;
	MPI_Datatype MPI_FLOAT2, MPI_FLOAT3, MPI_FLOAT4;
	MPI_Datatype MPI_DOUBLE2, MPI_DOUBLE3, MPI_DOUBLE4;

	MPI::Type_contiguous(2, MPI_INT, &MPI_INT2);
	MPI::Type_contiguous(3, MPI_INT, &MPI_INT3);
	MPI::Type_contiguous(4, MPI_INT, &MPI_INT4);

	MPI::Type_contiguous(2, MPI_FLOAT, &MPI_FLOAT2);
	MPI::Type_contiguous(3, MPI_FLOAT, &MPI_FLOAT3);
	MPI::Type_contiguous(4, MPI_FLOAT, &MPI_FLOAT4);

	MPI::Type_contiguous(2, MPI_DOUBLE, &MPI_DOUBLE2);
	MPI::Type_contiguous(3, MPI_DOUBLE, &MPI_DOUBLE3);
	MPI::Type_contiguous(4, MPI_DOUBLE, &MPI_DOUBLE4);

	MPI::Type_commit<int2>(&MPI_INT2);
	MPI::Type_commit<int3>(&MPI_INT3);
	MPI::Type_commit<int4>(&MPI_INT4);

	MPI::Type_commit<float2>(&MPI_FLOAT2);
	MPI::Type_commit<float3>(&MPI_FLOAT3);
	MPI::Type_commit<float4>(&MPI_FLOAT4);

	MPI::Type_commit<double2>(&MPI_DOUBLE2);
	MPI::Type_commit<double3>(&MPI_DOUBLE3);
	MPI::Type_commit<double4>(&MPI_DOUBLE4);
};

}