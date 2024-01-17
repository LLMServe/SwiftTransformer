#pragma once

#include <cstdint>
#include <mpi.h>
#include <nccl.h>

namespace st::util {
// Only type with fixed size is supported
class StDataType {
public:
    enum Type {
        INT8,
        INT32,
        INT64,
        FLOAT16,
        FLOAT32,
        FLOAT64,
        BOOL,
    };
    StDataType() = default;
    constexpr StDataType(Type type)
        : type(type)
    {
    }
    constexpr int64_t get_size() const
    {
        switch (type) {
        case INT8:
            return 1;
        case INT32:
            return 4;
        case INT64:
            return 8;
        case FLOAT16:
            return 2;
        case FLOAT32:
            return 4;
        case FLOAT64:
            return 8;
        case BOOL:
            return 1;
        }
        printf("[ERROR] Unsupported data type\n");
        exit(-1);
    }
    constexpr Type get_type() const { return type; }

    // TODO(sunyh): spaceship operator after C++20
    constexpr bool operator==(const StDataType& other) const { return type == other.type; }
    constexpr bool operator!=(const StDataType& other) const { return type != other.type; }

    constexpr ncclDataType_t get_nccl_type() const
    {
        switch (type) {
        case StDataType::INT8:
            return ncclInt8;
        case StDataType::INT32:
            return ncclInt32;
        case StDataType::INT64:
            return ncclInt64;
        case StDataType::FLOAT16:
            return ncclFloat16;
        case StDataType::FLOAT32:
            return ncclFloat32;
        case StDataType::FLOAT64:
            return ncclFloat64;
        case StDataType::BOOL:
            return ncclInt8;
        default:
            printf("[ERROR] Unsupported data type\n");
            exit(-1);
        }
    }

    constexpr MPI_Datatype get_mpi_type() const
    {
        switch (type) {
        case INT8:
            return MPI_INT8_T;
        case INT32:
            return MPI_INT32_T;
        case INT64:
            return MPI_INT64_T;
        case FLOAT16:
            return MPI_INT16_T; // MPI_FLOAT16_T is not supported, use MPI_INT16_T instead
        case FLOAT32:
            return MPI_FLOAT;
        case FLOAT64:
            return MPI_DOUBLE;
        case BOOL:
            return MPI_C_BOOL;
        }
        printf("[ERROR] Unsupported data type\n");
        exit(-1);
    }

private:
    Type type;
};

template<typename T>
StDataType stGetDataType(){
    if (std::is_same<T, int8_t>::value) {
        return StDataType(StDataType::INT8);
    } else if (std::is_same<T, int32_t>::value) {
        return StDataType(StDataType::INT32);
    } else if (std::is_same<T, int64_t>::value) {
        return StDataType(StDataType::INT64);
    } else if (std::is_same<T, half>::value) {
        return StDataType(StDataType::FLOAT16);
    } else if (std::is_same<T, float>::value) {
        return StDataType(StDataType::FLOAT32);
    } else if (std::is_same<T, double>::value) {
        return StDataType(StDataType::FLOAT64);
    } else if (std::is_same<T, bool>::value) {
        return StDataType(StDataType::BOOL);
    } else {
        printf("[ERROR] Unsupported data type\n");
        exit(-1);
    }
}

}