#ifndef FAKEMPI_H
#define FAKEMPI_H
#include <ctime>

typedef int MPI_Comm;
typedef int MPI_Group;

inline double
MPI_Wtime()
{
    return clock()/(1.0*CLOCKS_PER_SEC);
}

const int MPI_COMM_WORLD = 0;
const int MPI_SUCCESS = 0;
const int MPI_MAX_PROCESSOR_NAME = 128;

inline int
MPI_Comm_rank(MPI_Comm dummy, int* rank)
{
    *rank = 0;
    return 0;
}

inline int
MPI_Comm_size(MPI_Comm dummy, int* size)
{
    *size = 1;
    return 0;
}

inline int
MPI_Init(int* argc, char*** argv)
{
    return 0;
}

inline int
MPI_Finalize()
{
    return 0;
}

inline int
MPI_Comm_group(MPI_Comm comm, MPI_Group* group)
{
    return 0;
}

inline int
MPI_Group_incl(MPI_Group parent_group, int, int*, MPI_Group* group)
{
    return 0;
}

inline int
MPI_Comm_create(MPI_Comm parent_mpi_comm, MPI_Group group, MPI_Comm* temp_comm)
{
    return 0;
}

inline int
MPI_Group_free(MPI_Group* group)
{
    return 0;
}

inline int
MPI_Comm_free(MPI_Comm* commm)
{
    return 0;
}

inline int
MPI_Get_processor_name(char* name, int* name_len)
{
    char dummy[6] = "dummy";
    name = dummy;
    *name_len = 6;
    return 0;
}

inline int
MPI_Comm_split(MPI_Comm comm, int x, int y, MPI_Comm* other_comm)
{
    return 0;
}

#endif // FAKEMPI_H
