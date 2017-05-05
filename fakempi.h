#ifndef FAKEMPI_H
#define FAKEMPI_H

typedef int MPI_Comm;
typedef int MPI_Group;

#include <ctime>
double
MPI_Wtime()
{
    time_t timer = time(NULL);
    struct tm y2k = { 0 };

    y2k.tm_hour = 0;
    y2k.tm_min = 0;
    y2k.tm_sec = 0;
    y2k.tm_year = 100;
    y2k.tm_mon = 0;
    y2k.tm_mday = 1;

    return difftime(timer, mktime(&y2k));
}

const int MPI_COMM_WORLD = 0;
const int MPI_SUCCESS = 0;
const int MPI_MAX_PROCESSOR_NAME = 128;

int
MPI_Comm_rank(MPI_Comm dummy, int* rank)
{
    *rank = 0;
    return 0;
}
int
MPI_Comm_size(MPI_Comm dummy, int* size)
{
    *size = 1;
    return 0;
}
int
MPI_Init(int* argc, char*** argv)
{
    return 0;
}
int
MPI_Finalize()
{
    return 0;
}

int
MPI_Comm_group(MPI_Comm comm, MPI_Group* group)
{
    return 0;
}

int
MPI_Group_incl(MPI_Group parent_group, int, int*, MPI_Group* group)
{
    return 0;
}

int
MPI_Comm_create(MPI_Comm parent_mpi_comm, MPI_Group group, MPI_Comm* temp_comm)
{
    return 0;
}

int
MPI_Group_free(MPI_Group* group)
{
    return 0;
}

int
MPI_Comm_free(MPI_Comm* commm)
{
    return 0;
}

int
MPI_Get_processor_name(char* name, int* name_len)
{
    name = "dummy";
    *name_len = 5;
    return 0;
}

int
MPI_Comm_split(MPI_Comm comm, int x, int y, MPI_Comm* other_comm)
{
    return 0;
}

#endif // FAKEMPI_H
