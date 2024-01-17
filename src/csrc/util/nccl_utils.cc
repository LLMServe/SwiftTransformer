#include "nccl_utils.h"
#include <cstdio>

#define NCCL_CHECK(cmd)                                                                                           \
    do {                                                                                                          \
        ncclResult_t result = cmd;                                                                                \
        if (result != ncclSuccess) {                                                                              \
            printf("[ERROR] NCCL error %s:%d '%s' : %s\n", __FILE__, __LINE__, #cmd, ncclGetErrorString(result)); \
            exit(-1);                                                                                             \
        }                                                                                                         \
    } while (0)

namespace st::util {

void stNcclErrorCheck(ncclResult_t result, const char* func, const char* file, int line)
{
    if (result != ncclSuccess) {
        printf("[ERROR] NCCL error %s:%d '%s' : %s\n", file, line, func, ncclGetErrorString(result));
        exit(-1);
    }
}

void stNcclGetUniqueId(ncclUniqueId& nccl_id)
{
    NCCL_CHECK(ncclGetUniqueId(&nccl_id));
}

NcclComm stNcclInit(int64_t world_size, int64_t rank, const ncclUniqueId& nccl_id, cudaStream_t stream, bool real_init)
{
    NcclComm nccl_comm;
    nccl_comm.rank = rank;
    nccl_comm.size = world_size;
    nccl_comm.stream = stream;
    if (world_size == 1 || !real_init) {
        nccl_comm.comm = nullptr;
        return nccl_comm;
    }
    NCCL_CHECK(ncclCommInitRank(&nccl_comm.comm, nccl_comm.size, nccl_id, nccl_comm.rank));
    return nccl_comm;
}

void stNcclDestroy(NcclComm& nccl_comm)
{
    NCCL_CHECK(ncclCommDestroy(nccl_comm.comm));
}

void stNcclAllReduce(
    void* sendbuff,
    void* recvbuff,
    int64_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream)
{
    if (comm == nullptr) {
        return;
    }
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream));
}

void stNcclSend(
    void* buff,
    int64_t count,
    ncclDataType_t datatype,
    int64_t send_to,
    NcclComm comm,
    cudaStream_t stream)
{
    if (comm.comm == nullptr) {
        printf("[ERROR] NCCL comm is null\n");
        return;
    }

    if (send_to == comm.rank) {
        printf("[ERROR] Send rank and recv rank are the same\n");
        exit(-1);
    }

    NCCL_CHECK(ncclSend(buff, count, datatype, send_to, comm.comm, stream));
}

void stNcclRecv(
    void* buff,
    int64_t count,
    ncclDataType_t datatype,
    int64_t recv_from,
    NcclComm comm,
    cudaStream_t stream)
{
    if (comm.comm == nullptr) {
        printf("[ERROR] NCCL comm is null\n");
        return;
    }

    if (recv_from == comm.rank) {
        printf("[ERROR] Send rank and recv rank are the same\n");
        exit(-1);
    }

    NCCL_CHECK(ncclRecv(buff, count, datatype, recv_from, comm.comm, stream));
}

void stNcclSendRecv(
    void* sendbuff,
    void* recvbuff,
    int64_t count,
    ncclDataType_t datatype,
    int64_t send_rank,
    int64_t recv_rank,
    NcclComm comm,
    cudaStream_t stream)
{
    if (comm.comm == nullptr) {
        printf("[ERROR] NCCL comm is null\n");
        return;
    }

    if (send_rank == recv_rank) {
        printf("[ERROR] Send rank and recv rank are the same\n");
        exit(-1);
    }

    if (send_rank == comm.rank) {
        NCCL_CHECK(ncclSend(sendbuff, count, datatype, recv_rank, comm.comm, stream));
    } else if (recv_rank == comm.rank) {
        NCCL_CHECK(ncclRecv(recvbuff, count, datatype, send_rank, comm.comm, stream));
    } else {
        printf("[ERROR] Rank %ld is not involved in the send/recv\n", comm.rank);
        exit(-1);
    }
}
void stNcclBcast(
    void* buff,
    int64_t count,
    ncclDataType_t datatype,
    int64_t root,
    NcclComm comm,
    cudaStream_t stream)
{
    if (comm.comm == nullptr) {
        printf("[ERROR] NCCL comm is null\n");
        return;
    }
    NCCL_CHECK(ncclBcast(buff, count, datatype, root, comm.comm, stream));
}

}