#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "cublas_v2.h"
#include "thrust/sort.h"
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust\device_vector.h>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template <typename T>
struct inverse
{
	__host__ __device__
		T operator()(const T& x) const {
		return 1 / x;
	}
};

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#define IDX2C(i, j, ld) (((j)*(ld))+(i))
#define MAX_INPUTS 32

#define POPULATIONS 1
#define RULES_PER_NODE 1
#define NODES 32
#define POPULATION (1024*POPULATIONS)
//#define LAMBDA 0.9
#define ELITE_MEMBERS (POPULATION)
#define ELEMENTS_TO_MIGRATE (2*POPULATIONS)
#define MIGRATION_FREQUENCY 2

#define INDIVIDUALS_PER_BLOCK 2

#define MAX_CONNECTIVITY_DISTANCE 0.1

#define GENERATIONS 200
#define EXECUTIONS 1
#define LINK_MUTATION_PROB 0.001  
#define RULE_MUTATION_PROB 0.001

#define INDIVIDUAL_SIZE (NODES * MAX_INPUTS)

#define ROULETTE 0
#define ELITE 1

#define MOST 0
#define ABSOLUTE_REPRESSOR 1
#define JOINT_ACTIVATORS 2
#define JOINT_REPRESSORS 3

typedef struct
{
	char nodes[NODES];
	char links[NODES*MAX_INPUTS];
	char rules[RULES_PER_NODE*NODES];
} network;

__device__ const network INIT_NETWORK = {
	{ 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 0, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }
};
network INIT_NETWORK_HOST = {
	{ 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 0, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }
};
network GOAL_NETWORK_HOST = {
	{ 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
	{ 1, 0, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }
};

float LAMBDA_HOST[POPULATIONS] = { .9 };

__constant__ char GOAL_LINKS[NODES*MAX_INPUTS];
__constant__ char GOAL_NODES[NODES];
__constant__ char INIT_NODES[NODES];

__constant__ float LAMBDA_VALUES[POPULATIONS];

__constant__ float INIT_CONNECTIVITY;

float BEST_FITNESS_HOST = 1.0;

__device__ unsigned int WangHash(unsigned int a){
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ float generate(curandState* globalState)
{
	int ind = blockIdx.x*blockDim.x + threadIdx.x;

	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}


__device__ void generate_v2(curandState* globalState, float * values, unsigned int count)
{
	int ind = blockIdx.x*blockDim.x + threadIdx.x;

	curandState localState = globalState[ind];
	for (int i = 0; i < count; i++){
		values[i] = curand_uniform(&localState);
	}

	globalState[ind] = localState;

}

__device__ void generate_v3(curandState* globalState, int * values, unsigned int count, int max_value)
{
	int ind = blockIdx.x*blockDim.x + threadIdx.x;

	curandState localState = globalState[ind];
	for (int i = 0; i < count; i++){
		values[i] = (int)(curand_uniform(&localState) * max_value);
	}

	globalState[ind] = localState;

}


__global__ void setup_kernel(curandState * state)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int seed = (unsigned int)clock64();
	curand_init(seed, id, 0, &state[id]);
}

__global__ void setup_kernel_V2(curandState * state)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int seed = (unsigned int)clock64();
	curand_init((seed << 20) + id, 0, 0, &state[id]);
}

__global__ void setup_kernel_V3(curandState * state)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int seed = (unsigned int)clock64();
	curand_init(WangHash(seed) + id, 0, 0, &state[id]);
}
__global__ void setup_kernel_V4(curandState * state)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int seed = (unsigned int)clock64();
	curand_init(WangHash(seed), id, 0, &state[id]);
}
__global__ void setup_kernel_V5(curandState * state)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int seed = (unsigned int)clock64();
	curand_init(WangHash(seed) + id, id, 0, &state[id]);
}
/**
* Shuffles the elements of a given array.
* @param array Array to shuffle.
* @param n Elements in the array.
*/
__device__  void shuffle(char *array, int n, curandState *globalState)
{
	int i = n - 1;
	int j, temp;
	while (i > 0)
	{
		j = generate(globalState)*(i + 1);
		temp = array[i];
		array[i] = array[j];
		array[j] = temp;
		i = i - 1;
	}
}


/*
______     _       _   _
| ___ \   (_)     | | (_)
| |_/ / __ _ _ __ | |_ _ _ __   __ _
|  __/ '__| | '_ \| __| | '_ \ / _` |
| |  | |  | | | | | |_| | | | | (_| |
\_|  |_|  |_|_| |_|\__|_|_| |_|\__, |
__/ |
|___/
*/


__host__ __device__ void print_array(const char array[], int size)
{
	int i;
	//	printf("{");
	for (i = 0; i < size - 1; i++)
	{
		printf("%d,", array[i]);
	}
	printf("%d", array[size - 1]);
	printf("\n");
}
__host__ __device__ void print_array_f(const float array[], int size)
{
	int i;
	printf("[");
	for (i = 0; i < size; i++)
	{
		printf("%10f", array[i]);
	}
	printf("]\n");
}
__host__ void print_array_file(const char array[], int size, FILE *f)
{
	int i;
	fprintf(f, "{%d", array[0]);
	for (i = 1; i < size; i++)
	{
		fprintf(f, ",%d", array[i]);
	}
	fprintf(f, "};\n");
}


__device__ __host__ void print_network(network individual)
{
	printf("--\n");
	print_array(individual.nodes, NODES);
	print_array(individual.links, NODES * MAX_INPUTS);
	print_array(individual.rules, RULES_PER_NODE * NODES);
}
__host__ void print_network_file(network individual, FILE *f)
{
	fprintf(f, "\nNODES: ");
	print_array_file(individual.nodes, NODES, f);
	fprintf(f, "\nLINKS: ");
	print_array_file(individual.links, NODES * MAX_INPUTS, f);
	fprintf(f, "\nRULES: ");
	print_array_file(individual.rules, RULES_PER_NODE * NODES, f);
}

__device__ __host__ void print_population(network population[], int population_size)
{
	int i;
	printf("------------------- POPULATION -------------------\n");
	for (i = 0; i < population_size; ++i)
	{
		print_network(population[i]);
		//  printf("Fitness value: %f\n",evaluate_individual(&population[i], &GOAL_NETWORK_HOST, MAX_INPUTS, NODES, LAMBDA,0));
	}
	printf("---------------------------------------------------\n");
}
void inline print_progress(float progress, float best)
{
	if (progress < 1.0)
	{
		int barWidth = 50;
		int i;

		printf(" [");
		int pos = barWidth * progress;
		for (i = 0; i < barWidth; ++i)
		{
			if (i < pos) printf("%c", '=');
			else if (i == pos) printf("%c", '>');
			else printf("%c", ' ');
		}
		// printf("] %.1f%\r", progress * 100.0);
		printf("] %.2f -- Best: %.8f\r", progress * 100, best);

		fflush(stdout);
	}
}

void print_parameters(){
	unsigned int width = 35;
	printf("TOTAL POPULATION: %*d\n", width, POPULATION);
	printf("ISLANDS (POPULATIONS): %*d\n", width, POPULATIONS);
	printf("MIGRATIONS/ISLAND: %*d\n", width, ELEMENTS_TO_MIGRATE / POPULATIONS);
	printf("MIGRATION FREQUENCY: %*d\n", width, MIGRATION_FREQUENCY);
	//printf("LAMBDA: \t%*.3f\n", width, LAMBDA);
	printf("LINK MUTATION PROB: %*.3f\n", width, LINK_MUTATION_PROB);
	printf("RULE MUTATION PROB: %*.3f\n", width, RULE_MUTATION_PROB);
	printf("GENERATIONS: %*d\n", width, GENERATIONS);
}


/*
_   _           _         _        _
| \ | |         | |       | |      | |
|  \| | _____  _| |_   ___| |_ __ _| |_ ___
| . ` |/ _ \ \/ / __| / __| __/ _` | __/ _ \
| |\  |  __/>  <| |_  \__ \ || (_| | ||  __/
\_| \_/\___/_/\_\\__| |___/\__\__,_|\__\___|

*/
template <unsigned int individuals_per_block>
__device__ void apply_rules(char links[], int links_number, char rule, char nodes[], unsigned int node_index)
{

	__shared__ unsigned char input_count[3 * individuals_per_block*NODES];

	char a, count_index;

	char states[3] = { 0, 1, nodes[node_index] };

	char rules_to_states[4];

	for (char i = 0; i < links_number; i++)
	{
		a = (links[i] * nodes[i + (int)(node_index / NODES)*NODES]);

		count_index = (-(a - 1)*(3 * a + 4)) >> 1;

		input_count[count_index + node_index * 3]++;

	}

	__syncthreads();

	char x1 = (input_count[0 + node_index * 3] > input_count[1 + node_index * 3]) - (input_count[0 + node_index * 3] < input_count[1 + node_index * 3]); // (+ > -) - ( + < -)
	rules_to_states[0] = (-(x1 + 1)*(3 * x1 - 4)) >> 1;

	char x2 = input_count[1 + node_index * 3]>0;
	rules_to_states[1] = 2 - 2 * x2;

	char x3 = input_count[0 + node_index * 3]>1;
	rules_to_states[2] = 2 - x3;

	char x4 = input_count[1 + node_index * 3]>1;
	rules_to_states[3] = 2 - 2 * x4;

	nodes[node_index] = states[rules_to_states[rule]];

	__syncthreads();


	
	/*unsigned char input_count[3] = { 0, 0, 0 };

	char a, count_index;

	char states[3] = { 0, 1, nodes[node_index] };

	char rules_to_states[4];

	for (char i = 0; i < links_number; i++)
	{
		a = (links[i] * nodes[i + (int)(node_index / NODES)*NODES]);

		count_index = (-(a - 1)*(3 * a + 4)) >> 1;

		input_count[count_index]++;

	}

	__syncthreads();

	char x1 = (input_count[0] > input_count[1]) - (input_count[0] < input_count[1]); // (+ > -) - ( + < -)
	rules_to_states[MOST] = (-(x1 + 1)*(3 * x1 - 4)) >> 1;

	char x2 = input_count[1]>0;
	rules_to_states[ABSOLUTE_REPRESSOR] = 2 - 2 * x2;

	char x3 = input_count[0]>1;
	rules_to_states[JOINT_ACTIVATORS] = 2 - x3;

	char x4 = input_count[1]>1;
	rules_to_states[JOINT_REPRESSORS] = 2 - 2 * x4;

	unsigned char new_state = states[rules_to_states[rule]];

	nodes[node_index] = new_state;

	__syncthreads();
	*/

	/*unsigned char repressor_count, activator_count;
	char i;
	// ¿Merece la pena usar reducción?
	repressor_count = activator_count = 0;

	for (i = 0; i < links_number; i++)
	{
		if ((links[i] * nodes[i + (int)(node_index / NODES)*NODES])< 0)
		{
			repressor_count++;
		}
		else if ((links[i] * nodes[i + (int)(node_index / NODES)*NODES])> 0)
		{
			activator_count++;
		}
	}

	__syncthreads(); // TODO NECESARIO?

	switch (rule)
	{
	case MOST:

		if (repressor_count > activator_count)
		{
			nodes[node_index] = 0;
		}
		else if (activator_count > repressor_count)
		{
			nodes[node_index] = 1;
		}
		break;

	case ABSOLUTE_REPRESSOR:
		if (nodes[node_index] == 1 && repressor_count >= 1){
			nodes[node_index] = 0;
		}
		break;
	case JOINT_ACTIVATORS:
		if (nodes[node_index] == 0 && activator_count > 1){
			nodes[node_index] = 1;
		}
		break;
	case JOINT_REPRESSORS:
		if (nodes[node_index] == 1 && repressor_count > 1){
			nodes[node_index] = 0;
		}
		break;
	default:
		break;
	}

	__syncthreads();
	*/
}





/*
______ _ _
|  ___(_) |
| |_   _| |_ _ __   ___  ___ ___
|  _| | | __| '_ \ / _ \/ __/ __|
| |   | | |_| | | |  __/\__ \__ \
\_|   |_|\__|_| |_|\___||___/___/


*/
template <unsigned int elementsPerThread>
__global__ void link_distance(network population[], float *distances, int size, char goal_links[]){

	const unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int individual = (index*elementsPerThread) / INDIVIDUAL_SIZE;
	const unsigned int link = index % INDIVIDUAL_SIZE;
	const unsigned int population_index = individual / (POPULATION / POPULATIONS);

	if (individual < POPULATION){
		float sum = 0;
		for (int i = 0; i < elementsPerThread; i++){
			float distance = (population[individual].links[link + (elementsPerThread*i)] != goal_links[link + (elementsPerThread*i)]);
			sum += distance;
		}

		sum *= (1 - LAMBDA_VALUES[population_index]) / (NODES*MAX_INPUTS);
		distances[IDX2C(individual, link, POPULATION)] = sum;
	}
}

template <unsigned int individuals_per_block>
__global__ void node_distance(network population[], float *distances, int size, char init_nodes[], char goal_nodes[]){

	//const unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int offset = blockIdx.x * individuals_per_block;
	const unsigned int individual = (threadIdx.x / NODES) + offset;
	const unsigned int population_index = individual / (POPULATION / POPULATIONS);
	const unsigned char node = threadIdx.x % NODES;

	//	const unsigned int link_local_index = (node*MAX_INPUTS) + (threadIdx.x%individuals_per_block)*INDIVIDUAL_SIZE;

	__shared__ char nodes[individuals_per_block*NODES];
	//__shared__ char individual_links[blockSize*MAX_INPUTS];

	char node_inputs[MAX_INPUTS];

	if (individual < POPULATION){

		nodes[threadIdx.x] = init_nodes[node];

		__syncthreads();

		const unsigned char rule = population[individual].rules[node];

		//TODO En la primera iteracion puedo calcular directamente represores y activadores
		for (int i = 0; i < MAX_INPUTS; i++){
			node_inputs[i] = population[individual].links[node*MAX_INPUTS + i];
		}


		for (int i = 0; i < 5; i++){
			apply_rules<individuals_per_block>(node_inputs, MAX_INPUTS, rule, nodes, threadIdx.x);
		}

		float distance = (nodes[threadIdx.x] != goal_nodes[node]);

		distance *= LAMBDA_VALUES[population_index] / NODES;
		distances[IDX2C(individual, node, POPULATION)] = distance;
	}
}


cublasStatus_t sum_link_distances(const float* A, const float* d_x, float* d_y, const int row, const int col, cublasHandle_t handle){

	// level 2 calculation y = alpha * A * x + beta * y
	float alf = 1.f;
	float beta = 0.f;

	return cublasSgemv(handle, CUBLAS_OP_N, col, row, &alf, A, col, d_x, 1, &beta, d_y, 1);//swap col and row
}

cublasStatus_t sum_fitness(const float* fitness, float* result, cublasHandle_t handle){

	return cublasSasum(handle, POPULATION, fitness, 1, result);
}

cublasStatus_t sum_node_distances(const float* A, const float* d_x, float* d_y, const int row, const int col, cublasHandle_t handle){

	// level 2 calculation y = alpha * A * x + beta * y
	float alf = 1.f;
	float beta = 0.f;

	return cublasSgemv(handle, CUBLAS_OP_N, col, row, &alf, A, col, d_x, 1, &beta, d_y, 1);//swap col and row

}
cublasStatus_t pop_fitness(const float* link_distances, const float* node_distances, float* fitness, cublasHandle_t handle){

	// level 2 calculation y = alpha * A * x + beta * y
	float alf = 1.f;
	float beta = 1.f;
	return cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, POPULATION, 1, &alf, link_distances, POPULATION, &beta, node_distances, POPULATION, fitness, POPULATION);

}

/*
_____                           _         _           _ _       _     _             _
|  __ \                         | |       (_)         | (_)     (_)   | |           | |
| |  \/ ___ _ __   ___ _ __ __ _| |_ ___   _ _ __   __| |___   ___  __| |_   _  __ _| |___
| | __ / _ \ '_ \ / _ \ '__/ _` | __/ _ \ | | '_ \ / _` | \ \ / / |/ _` | | | |/ _` | / __|
| |_\ \  __/ | | |  __/ | | (_| | ||  __/ | | | | | (_| | |\ V /| | (_| | |_| | (_| | \__ \
\____/\___|_| |_|\___|_|  \__,_|\__\___| |_|_| |_|\__,_|_| \_/ |_|\__,_|\__,_|\__,_|_|___/

*/
__device__ void generate_rules(char rules[], int size, curandState * globalState)
{
	int i, rule;
	float randoms[RULES_PER_NODE*NODES];
	generate_v2(globalState, randoms, RULES_PER_NODE*NODES);
	for (i = 0; i < size; i++)
	{
		rule = randoms[i] * 4;
		rules[i] = rule; /* Four possible rules */
	}
}


__global__ void generate_individual(network population[], int nodes, int max_inputs, int rules_per_node, curandState* globalState)
{

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	// TODO ¿individuo en local?
	if (i < POPULATION){
		population[i] = INIT_NETWORK;
		shuffle(population[i].links, max_inputs * nodes, globalState);
		generate_rules(population[i].rules, rules_per_node * nodes, globalState);
	}

}

__global__ void mutation(network *population, int nodes, int max_inputs_per_node, int rules_per_node, float link_mut_prob, float rule_mut_prob, curandState *globalState)
{

	unsigned int individual_index = (blockIdx.x*blockDim.x + threadIdx.x);
	//TODO ¿Se están guardando los arrays en memoria local por culpa del indexado?

	char rules[] = { 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2 };
	char links[] = { 0, 1, 1, -1, 0, -1 };

	const int rand_size = 6;

	char link_to_change, rule_to_change;
	int link_index, rule_index;
	int new_link_index, new_rule_index;
	double random_prob;

	float randoms[rand_size];
	generate_v2(globalState, randoms, rand_size);	// V2 minimizes memory R/W

	if (individual_index < POPULATION){
		//network individual = population[individual_index];
		//if (threadIdx.x < (POPULATION / 2)){
		/* LINKS MUTATION */
		random_prob = randoms[0];		// TODO considerar generar arrays de random, para minimizar escrituras en memoria cada vez
		if (random_prob <= link_mut_prob)
		{
			link_index = randoms[1] * (nodes * max_inputs_per_node);
			link_to_change = population[individual_index].links[link_index];
			new_link_index = randoms[2] * 2 + (link_to_change + 1) * 2;
			population[individual_index].links[link_index] = links[new_link_index];
		}
		//	}
		//	else{
		random_prob = randoms[3];
		if (random_prob <= rule_mut_prob)
		{
			rule_index = randoms[4] * rules_per_node * nodes;
			rule_to_change = population[individual_index].rules[rule_index];
			new_rule_index = randoms[5] * 3 + rule_to_change * 3;
			population[individual_index].rules[rule_index] = rules[new_rule_index];
		}
		//	}
	}

}


/*	 _   _ _   _ _
| | | | | (_) |
| | | | |_ _| |___
| | | | __| | / __|
| |_| | |_| | \__ \
\___/ \__|_|_|___/

*/

template <unsigned int blockSize>
__device__ void calculeConnectivity_v7(char links[], int size, float *connectivity){

	const unsigned int tid = threadIdx.x;

	__shared__ char sdata[INDIVIDUAL_SIZE / 2];

	int mySum;

	if (tid < size){
		mySum = links[tid] != 0;
	}

	if (tid + blockSize < size){
		mySum += (int)(links[tid + blockSize] != 0);
	}

	sdata[tid] = mySum;
	__syncthreads();

	// do reduction in shared mem
	if ((blockSize >= 512) && (tid < 256))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 256];
	}

	__syncthreads();

	if ((blockSize >= 256) && (tid < 128))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 128];
	}

	__syncthreads();

	if ((blockSize >= 128) && (tid < 64))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 64];
	}

	__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockSize >= 64){
			mySum += sdata[tid + 32];
		}
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down(mySum, offset);
		}
	}
#else
	// fully unroll reduction within a single warp
	if ((blockSize >= 64) && (tid < 32))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 32];
	}

	__syncthreads();

	if ((blockSize >= 32) && (tid < 16))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 16];
	}

	__syncthreads();

	if ((blockSize >= 16) && (tid < 8))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 8];
	}

	__syncthreads();

	if ((blockSize >= 8) && (tid < 4))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 4];
	}

	__syncthreads();

	if ((blockSize >= 4) && (tid < 2))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 2];
	}

	__syncthreads();

	if ((blockSize >= 2) && (tid < 1))
	{
		sdata[tid] = mySum = mySum + sdata[tid + 1];
	}

	__syncthreads();
#endif

	if (tid == 0){
		*connectivity = ((float)mySum) / size;
	}
	__syncthreads();	// ¿Necesario?
}

__global__ void sus_selection_v1(float population_fitness[], float total_fitness, int parents, int indexes[], curandState *globalState)
{
	// Calculate distance between the pointers
	float pointer_distance = total_fitness / parents;
	// Pick random number between 0 and p
	float start = generate(globalState) * pointer_distance;
	int index = 0;
	float sum = population_fitness[index];
	int i;
	float pointer;
	for (i = 0; i < parents / POPULATIONS; i++)
	{
		pointer = start + i*pointer_distance;
		if (sum >= pointer)
		{
			indexes[i] = index;
		}
		else
		{
			for (++index; index < POPULATION; index++)
			{
				sum += population_fitness[index];
				if (sum >= pointer)
				{
					indexes[i] = index;
					break;
				}
			}
		}
	}
}

__global__ void min(network population[], float* elements, int size, int generation, float *BEST_INDIVIDUAL_FITNESS, network * BEST_INDIVIDUAL){

	float min = 1;
	int index = 0;
	for (int i = 0; i<size; i++){
		if (min>elements[i]){
			min = elements[i];
			index = i;
		}
	}
	if (min < *(BEST_INDIVIDUAL_FITNESS)){
		*BEST_INDIVIDUAL_FITNESS = min;
		*BEST_INDIVIDUAL = population[index];
		//	print_network(population[index]);
	}

	//print_network(population[index]);
}

void min_v2(network population[], thrust::device_ptr<float> fitness, thrust::device_ptr<float> min_ptr, int generation, FILE *f, float * best_fitness, network * best_individual){

	float min = min_ptr[0];
	int position = thrust::distance(fitness, min_ptr);

	if (min < *(best_fitness)){
		*best_fitness = min;

		HANDLE_ERROR(
			cudaMemcpy(best_individual, population + position, sizeof(network), cudaMemcpyDeviceToHost)
			);

	}

	fprintf(f, "%d,%.8f\n", generation, *best_fitness);

}

void sort_population(thrust::device_ptr<float> keys, thrust::device_ptr<int> indexes){

	for (int i = 0; i < (POPULATIONS); i++){
		thrust::sort_by_key(keys + i*POPULATION / POPULATIONS, keys + (i + 1)*POPULATION / POPULATIONS, indexes + i*POPULATION / POPULATIONS);
	}

}

__global__ void sequence(int * indexes){
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	if (index < POPULATION){
		indexes[index] = index % (POPULATION / POPULATIONS);
	}
}


/*
_____
/  __ \
| /  \/_ __ ___  ___ ___  _____   _____ _ __
| |   | '__/ _ \/ __/ __|/ _ \ \ / / _ \ '__|
| \__/\ | | (_) \__ \__ \ (_) \ V /  __/ |
\____/_|  \___/|___/___/\___/ \_/ \___|_|

*/

__global__ void generateLinkCrossoverIndexes(int *indexes, curandState *globalState){
	int index = threadIdx.x + blockDim.x *blockIdx.x;
	int N = ELITE_MEMBERS / 2;
	if (index < N){
		indexes[index] = generate(globalState)*NODES*MAX_INPUTS;
	}
}
__global__ void generateRuleCrossoverIndexes(int *indexes, curandState *globalState){
	int index = threadIdx.x + blockDim.x *blockIdx.x;
	int N = ELITE_MEMBERS / 2;
	if (index < N){
		indexes[index] = generate(globalState)*RULES_PER_NODE*NODES;
	}
}

template <unsigned int blockSize>
__global__ void crossover(network population[], int *indexes_best, int *link_crossover_indexes){		//Considerar usar memoria constante aqui

	const unsigned int population_index = (blockIdx.x * 2) / (POPULATION / POPULATIONS);
	const unsigned int offset = POPULATION / POPULATIONS;

	int index_parent_1 = indexes_best[blockIdx.x * 2] + offset*population_index;
	int index_parent_2 = indexes_best[blockIdx.x * 2 + 1] + offset*population_index;

	const unsigned int tid = threadIdx.x;

	const int size = NODES*MAX_INPUTS;
	const int rules = RULES_PER_NODE*NODES;

	// Replace parents!
	int index_replacement_1 = index_parent_1;
	int index_replacement_2 = index_parent_2;

	//TODO ¿Guardar una pequeña variable con el indice de su enlace y ahorrar una lectura?

	__shared__ char links_child_1[size];
	__shared__ char links_child_2[size];

	__shared__ unsigned char rules_child_2[rules];

	int link_crossover_index = link_crossover_indexes[blockIdx.x];
	int rule_crossover_index = (link_crossover_index / NODES) + 1;

	/** Copy the children's links **/

	if (tid < size){
		if (tid < link_crossover_index){
			links_child_1[tid] = population[index_parent_1].links[tid];
			links_child_2[tid] = population[index_parent_2].links[tid];
		}
		else{
			links_child_1[tid] = population[index_parent_2].links[tid];
			links_child_2[tid] = population[index_parent_1].links[tid];
		}
	}

	if (tid + blockSize < size){
		if (tid + blockSize < link_crossover_index){
			links_child_1[tid + blockSize] = population[index_parent_1].links[tid + blockSize];
			links_child_2[tid + blockSize] = population[index_parent_2].links[tid + blockSize];
		}
		else{
			links_child_1[tid + blockSize] = population[index_parent_2].links[tid + blockSize];
			links_child_2[tid + blockSize] = population[index_parent_1].links[tid + blockSize];
		}
	}

	if (tid < rules){
		if (tid < rule_crossover_index){
			//	rules_child_1[tid] = population[index_parent_1].rules[tid];
			//	rules_child_2[tid] = population[index_parent_2].rules[tid];
		}
		else{
			//	rules_child_1[tid] = population[index_parent_2].rules[tid];
			rules_child_2[tid] = population[index_parent_1].rules[tid];
		}
	}

	__syncthreads();


	__shared__ float child_connectivity_1, child_connectivity_2;

	/** Child 1 **/
	calculeConnectivity_v7<blockSize>(links_child_1, NODES*MAX_INPUTS, &child_connectivity_1);

	if (fabsf(child_connectivity_1 - INIT_CONNECTIVITY) < MAX_CONNECTIVITY_DISTANCE){
		// Links
		if (tid > link_crossover_index){
			population[index_replacement_1].links[tid] = links_child_1[tid];
		}
		if (blockSize >= 512 && ((tid + 512) > link_crossover_index)){
			population[index_replacement_1].links[tid + 512] = links_child_1[tid + 512];
		}

		// Rules
		if (tid < rules && tid >= rule_crossover_index){
			population[index_replacement_1].rules[tid] = population[index_parent_2].rules[tid];
		}
	}
	// Sync?

	/** Child 2 **/
	calculeConnectivity_v7<blockSize>(links_child_2, NODES*MAX_INPUTS, &child_connectivity_2);

	if (fabsf(child_connectivity_2 - INIT_CONNECTIVITY) < MAX_CONNECTIVITY_DISTANCE){
		// Links
		if (tid > link_crossover_index){
			population[index_replacement_2].links[tid] = links_child_2[tid];
		}
		if (blockSize >= 512 && ((tid + 512) > link_crossover_index)){
			population[index_replacement_2].links[tid + 512] = links_child_2[tid + 512];
		}

		// Rules
		if (tid < rules && tid >= rule_crossover_index){
			population[index_replacement_2].rules[tid] = rules_child_2[tid];
		}
	}

}

__global__ void setup_indexes(int* indexes_best, thrust::device_ptr<int> positions){

	int offset = POPULATION / POPULATIONS;
	int index = (threadIdx.x + blockDim.x*blockIdx.x);
	unsigned int parents_per_population = ELITE_MEMBERS / POPULATIONS;
	int population = index / parents_per_population;

	if (index < (ELITE_MEMBERS)){
		indexes_best[index] = positions[population*offset + index%parents_per_population];
	}

}

void total_fitness(thrust::device_ptr<float> keys, thrust::device_ptr<float> total_fitness){

	inverse<float> unary_op;
	thrust::plus<float> binary_op;
	float init = 0;

	for (int i = 0; i < (POPULATIONS); i++){
		total_fitness[i] = thrust::transform_reduce(keys + i*POPULATION / POPULATIONS, keys + (i + 1)*POPULATION / POPULATIONS, unary_op, init, binary_op);
	}

}



__global__ void roulette_selection(float pop_fitness[], int indexes[], thrust::device_ptr<float> total_fitness, curandState *globalState){

	const unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	const unsigned int population_index = index / (POPULATION / POPULATIONS);
	//	const unsigned int individual_index = index % (POPULATION/POPULATIONS);
	const unsigned int pop_offset = POPULATION / POPULATIONS;

	if (index < POPULATION){
		float random = generate(
			globalState) * total_fitness[population_index];
		int pick = pop_offset*population_index;
		double offset = 0;
		float individual_fitness = 1 / pop_fitness[pick];
		while (random > offset){
			offset += powf(individual_fitness, 3);
			pick++;
		}
		indexes[index] = pick % (POPULATION / POPULATIONS);
	}

}

__global__ void sus_selection(float pop_fitness[], int indexes[], thrust::device_ptr<float> total_fitness, curandState *globalState){

	const unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	const unsigned int population_index = index / (POPULATION / POPULATIONS);
	//	const unsigned int individual_index = index % (POPULATION/POPULATIONS);
	const unsigned int pop_offset = POPULATION / POPULATIONS;

	__shared__ float randoms[POPULATIONS];

	if (index < POPULATION && (index % POPULATIONS) == 0){
		randoms[population_index] = generate(globalState);
	}

	__syncthreads();

	// Tener un random por  poblacion
	if (index < POPULATION){
		double p = total_fitness[population_index] / (POPULATION / POPULATIONS);
		int pick = pop_offset*population_index;
		double start = randoms[population_index] * p;
		double sum = 1 / pop_fitness[pick];
		double pointer = start + (index % (POPULATION / POPULATIONS)) * p;

		if (sum >= pointer){
			indexes[index] = pick;
		}
		else{
			for (++pick; pick < POPULATION; pick++){
				sum += 1 / pop_fitness[pick];
				if (sum >= pointer){
					indexes[index] = pick;
					break;
				}
			}
		}

	}

}



__global__ void migrate(network population[], int indexes_best[]){
	//One thread per individual
	//int index = threadIdx.x + blockDim.x*blockIdx.x;
	int index = blockIdx.x;
	const unsigned int tid = threadIdx.x;
	unsigned int elements_per_population = (ELEMENTS_TO_MIGRATE / POPULATIONS);
	unsigned int pop_index = index / elements_per_population;
	unsigned int offset = (ELITE_MEMBERS / POPULATIONS);
	unsigned int offset_elements = (POPULATION / POPULATIONS) * 2 - elements_per_population;
	unsigned int links_per_thread = INDIVIDUAL_SIZE / blockDim.x;

	if (index < ELEMENTS_TO_MIGRATE){
		unsigned int id_best = pop_index*offset + index%elements_per_population;
		unsigned int id_worst = (id_best + offset_elements) % ELITE_MEMBERS;
		unsigned int index_best = indexes_best[id_best];
		unsigned int index_worst = indexes_best[id_worst];

		for (int i = 0; i < links_per_thread; i++){
			population[index_worst + ((pop_index + 1)*POPULATION / POPULATIONS) % POPULATION].links[tid*links_per_thread + i] = population[(pop_index*POPULATION / POPULATIONS) + index_best].links[tid*links_per_thread + i];
		}
		if (threadIdx.x < NODES){		 // VS 32-63 para nodos? TODO
		//	population[index_worst + ((pop_index + 1)*POPULATION / POPULATIONS) % POPULATION].nodes[tid] = population[(pop_index*POPULATION / POPULATIONS) + index_best].nodes[tid];
			population[index_worst + ((pop_index + 1)*POPULATION / POPULATIONS) % POPULATION].rules[tid] = population[(pop_index*POPULATION / POPULATIONS) + index_best].rules[tid];
		}
		//	printf("\n%d (%d) <- %d (%d)\n", id_best, (pop_index+1)%ELITE_MEMBERS, id_worst, pop_index);
		// memcpy(&population[index_worst + ((pop_index+1)*POPULATION/POPULATIONS)%POPULATION], &population[(pop_index*POPULATION/POPULATIONS) + index_best], sizeof(network));
		// Links


	}
}

void compute_fitness(network * d_population, float * link_distances, float * d_x_links, float * link_fitness, cublasHandle_t handle, float * node_distances, float * d_x_nodes, float * node_fitness, float * d_current_fitness, char goal_links[], char init_nodes[], char goal_nodes[]){
	link_distance<1> << <(POPULATION*NODES*MAX_INPUTS + 32 * 4 - 1) / (32 * 4), 32 * 4 >> >(d_population, link_distances, POPULATION, goal_links);
	gpuErrchk(cudaPeekAtLastError());

	sum_link_distances(link_distances, d_x_links, link_fitness, INDIVIDUAL_SIZE, POPULATION, handle);
	gpuErrchk(cudaPeekAtLastError());

	node_distance <4> << <(POPULATION*NODES + 4 * 32 - 1) / (32 * 4), 32 * 4 >> >(d_population, node_distances, POPULATION, init_nodes, goal_nodes);
	gpuErrchk(cudaPeekAtLastError());

	sum_node_distances(node_distances, d_x_nodes, node_fitness, NODES, POPULATION, handle);
	gpuErrchk(cudaPeekAtLastError());

	pop_fitness(link_fitness, node_fitness, d_current_fitness, handle);
	gpuErrchk(cudaPeekAtLastError());
}

void elite_selection(network * d_population, float * link_distances, float * d_x_links, float * link_fitness, cublasHandle_t handle,
	float * node_distances, float * d_x_nodes, float * node_fitness, float * d_current_fitness, thrust::device_ptr<float> device_ptr_fitness,
	int * d_indexes, thrust::device_ptr<int> dev_indexes, int * d_indexes_best){

	sort_population(device_ptr_fitness, dev_indexes);
	gpuErrchk(cudaPeekAtLastError());

	//if (ELITE_MEMBERS != POPULATION){
	setup_indexes << <((ELITE_MEMBERS + 1023) * 20) / 1024, ELITE_MEMBERS / 16 >> >(d_indexes_best, dev_indexes);
	gpuErrchk(cudaPeekAtLastError());
	//}

}

void roulette_sel(network * d_population, float * link_distances, float * d_x_links, float * link_fitness, cublasHandle_t handle,
	float * node_distances, float * d_x_nodes, float * node_fitness, float * d_current_fitness, thrust::device_ptr<float> device_ptr_fitness,
	int * d_indexes, thrust::device_ptr<int> dev_indexes, int * d_indexes_best, curandState* globalState, thrust::device_ptr<float> total_pop_fitness){

	total_fitness(device_ptr_fitness, total_pop_fitness);

	roulette_selection << <(POPULATION + 512 + 1) / 512, 512 >> >(d_current_fitness, d_indexes_best, total_pop_fitness, globalState);
}

void migrate(unsigned int current_generation, unsigned int migration_freq, network * d_population, int * d_indexes_best){
	if (POPULATIONS > 1 && (current_generation%migration_freq) == 0){
		migrate << <ELEMENTS_TO_MIGRATE, 256 >> >(d_population, d_indexes_best);
		gpuErrchk(cudaPeekAtLastError());
	}
}

network population[POPULATION];

int main(void) {

	for (int e = 0; e < EXECUTIONS; e++){


		//TIME

		float time;
		cudaEvent_t start, stop;

		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventRecord(start));

		FILE *f;

		cudaError_t cudastat;
		cublasStatus_t stat;

		cublasHandle_t handle;
		stat = cublasCreate(&handle);

		float *d_current_fitness;		// Holds total population fitness for each generation

		float* link_fitness;		// Link fitness of each individual
		float* node_fitness;		// Node fitness of each individual

		float * d_BEST_INDIVIDUAL_FITNESS;
		network * d_BEST_INDIVIDUAL;
		network h_BEST_INDIVIDUAL;
		float h_BEST_INDIVIDUAL_FITNESS_HOST = 1.0;
		float aux = 1.0;

		float * d_x_links, *d_x_nodes;	// Array to hold the whether a link/node exists or not in the goal network
		int *d_indexes_best, *d_indexes_worst;	// Indexes for the best and worst individuals
		float *node_distances, *link_distances;	// Node and link distances PER NETWORK

		network *d_population;	// Network population

		curandState* devStates;

		int null_links = thrust::count(INIT_NETWORK_HOST.links, INIT_NETWORK_HOST.links + NODES*MAX_INPUTS, 0);
		gpuErrchk(cudaPeekAtLastError());
		const float init_connectivity = (NODES * MAX_INPUTS - (float)null_links) / (NODES * MAX_INPUTS);
		HANDLE_ERROR(
			cudaMemcpyToSymbol(INIT_CONNECTIVITY, &init_connectivity, sizeof(float)));

		HANDLE_ERROR(
			cudaMemcpyToSymbol(GOAL_LINKS, &GOAL_NETWORK_HOST.links, sizeof(char)*NODES*MAX_INPUTS));
		HANDLE_ERROR(
			cudaMemcpyToSymbol(GOAL_NODES, &GOAL_NETWORK_HOST.nodes, sizeof(char)*NODES));
		HANDLE_ERROR(
			cudaMemcpyToSymbol(INIT_NODES, &INIT_NETWORK_HOST.nodes, sizeof(char)*NODES));

		char buf[0x100];
		_snprintf(buf, sizeof(buf), "P-Sexec%d-%s_pob%dpops%d_MIGRs%d_gen%dfreq%d.csv", e, "ELITE", POPULATION, POPULATIONS, ELEMENTS_TO_MIGRATE / POPULATIONS, GENERATIONS, MIGRATION_FREQUENCY);

		f = fopen(buf, "w");
		if (f == NULL)
		{
			printf("Error opening file!\n");
			exit(1);
		}


		/*
		_                  __ _ _                             _       _
		| |                / _(_) |                           | |     | |
		___  ___| |_ _   _ _ __   | |_ _| |_ _ __   ___  ___ ___    __| | __ _| |_ __ _
		/ __|/ _ \ __| | | | '_ \  |  _| | __| '_ \ / _ \/ __/ __|  / _` |/ _` | __/ _` |
		\__ \  __/ |_| |_| | |_) | | | | | |_| | | |  __/\__ \__ \ | (_| | (_| | || (_| |
		|___/\___|\__|\__,_| .__/  |_| |_|\__|_| |_|\___||___/___/  \__,_|\__,_|\__\__,_|
		| |
		|_|
		*/



		HANDLE_ERROR(
			cudaMemcpyToSymbol(LAMBDA_VALUES, &LAMBDA_HOST, sizeof(float)*POPULATIONS));

		int * d_indexes;
		HANDLE_ERROR(
			cudaMalloc(&d_indexes, POPULATION*sizeof(int)));

		thrust::device_ptr<int> dev_indexes = thrust::device_pointer_cast(d_indexes);

		HANDLE_ERROR(
			cudaMalloc(&d_current_fitness, POPULATION*sizeof(float)));

		char * d_goal_links;
		HANDLE_ERROR(
			cudaMalloc(&d_goal_links, INDIVIDUAL_SIZE*sizeof(char)));
		HANDLE_ERROR(
			cudaMemcpy(d_goal_links, &GOAL_NETWORK_HOST.links, sizeof(char)*INDIVIDUAL_SIZE, cudaMemcpyHostToDevice));

		char * d_init_nodes;
		HANDLE_ERROR(
			cudaMalloc(&d_init_nodes, INDIVIDUAL_SIZE*sizeof(char)));
		HANDLE_ERROR(
			cudaMemcpy(d_init_nodes, &INIT_NETWORK_HOST.nodes, sizeof(char)*NODES, cudaMemcpyHostToDevice));

		char * d_goal_nodes;
		HANDLE_ERROR(
			cudaMalloc(&d_goal_nodes, NODES*sizeof(char)));
		HANDLE_ERROR(
			cudaMemcpy(d_goal_nodes, &GOAL_NETWORK_HOST.nodes, sizeof(char)*NODES, cudaMemcpyHostToDevice));


		float * total_fitness;
		HANDLE_ERROR(
			cudaMalloc(&total_fitness, POPULATIONS*sizeof(float)));
		thrust::device_ptr<float> dev_total_fitness = thrust::device_pointer_cast(total_fitness);

		HANDLE_ERROR(
			cudaMalloc(&d_BEST_INDIVIDUAL_FITNESS, sizeof(float)));
		HANDLE_ERROR(
			cudaMalloc(&d_BEST_INDIVIDUAL, sizeof(network)));


		HANDLE_ERROR(
			cudaMemcpy(d_BEST_INDIVIDUAL_FITNESS, &aux, sizeof(float), cudaMemcpyHostToDevice));

		thrust::device_ptr<float> device_ptr_fitness = thrust::device_pointer_cast(d_current_fitness);
		thrust::device_ptr<float> min_ptr = thrust::min_element(device_ptr_fitness, device_ptr_fitness + POPULATION);

		float* x_links = new float[NODES*MAX_INPUTS];

		for (int i = 0; i < NODES*MAX_INPUTS; i++)
		{
			x_links[i] = 1;//(1-LAMBDA)/(NODES*MAX_INPUTS);
		}

		float* x_nodes = new float[NODES];

		for (int i = 0; i < NODES; i++)
		{
			x_nodes[i] = 1;//(LAMBDA)/(NODES);
		}


		HANDLE_ERROR(
			cudaMalloc(&link_fitness, POPULATION*sizeof(float)));
		HANDLE_ERROR(
			cudaMalloc(&node_fitness, POPULATION*sizeof(float)));
		HANDLE_ERROR(
			cudaMalloc(&d_x_links, NODES*MAX_INPUTS*sizeof(float)));
		HANDLE_ERROR(
			cudaMemcpy(d_x_links, x_links, NODES*MAX_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

		HANDLE_ERROR(
			cudaMalloc(&d_x_nodes, NODES*sizeof(float)));
		HANDLE_ERROR(
			cudaMemcpy(d_x_nodes, x_nodes, NODES*sizeof(float), cudaMemcpyHostToDevice));


		HANDLE_ERROR(
			cudaMalloc(&d_indexes_best, ELITE_MEMBERS*sizeof(int)));
		HANDLE_ERROR(
			cudaMalloc(&d_indexes_worst, ELITE_MEMBERS*sizeof(int)));

		HANDLE_ERROR(
			cudaMalloc(&node_distances, POPULATION * NODES * sizeof(float)));
		HANDLE_ERROR(
			cudaMalloc(&link_distances, POPULATION * NODES * MAX_INPUTS * sizeof(float)));

		//*BEST_INDIVIDUAL_FITNESS_HOST = *aux;
		/*
		_____                 _            _
		|  __ \               | |          | |
		___ _   _| |__) |__ _ _ __   __| |  ___  ___| |_ _   _ _ __
		/ __| | | |  _  // _` | '_ \ / _` | / __|/ _ \ __| | | | '_ \
		| (__| |_| | | \ \ (_| | | | | (_| | \__ \  __/ |_| |_| | |_) |
		\___|\__,_|_|  \_\__,_|_| |_|\__,_| |___/\___|\__|\__,_| .__/
		| |
		|_|
		*/

		HANDLE_ERROR(
			cudaMalloc(&devStates, POPULATION * 1024 * sizeof(curandState)));

		print_parameters();

		setup_kernel_V3 << < POPULATION * 2, 512 >> > (devStates);
		gpuErrchk(cudaPeekAtLastError());

		HANDLE_ERROR(
			cudaMalloc(&d_population, POPULATION*sizeof(network)))
			;

		generate_individual << <(POPULATION + 1023) / 512, 512 >> >(d_population, NODES, MAX_INPUTS, RULES_PER_NODE, devStates);
		gpuErrchk(cudaPeekAtLastError());

		const int individuals_block_ns = INDIVIDUALS_PER_BLOCK;

		int *link_crossover_indexes, *rule_crossover_indexes;
		HANDLE_ERROR(
			cudaMalloc(&link_crossover_indexes, (ELITE_MEMBERS / 2)*sizeof(int)));
		HANDLE_ERROR(
			cudaMalloc(&rule_crossover_indexes, (ELITE_MEMBERS / 2)*sizeof(int)));

		float* h_fit = (float*)malloc(POPULATION*sizeof(float));


		for (int i = 0; i < GENERATIONS && h_BEST_INDIVIDUAL_FITNESS_HOST != 0; i++){

			//next_state << <(POPULATION + individuals_block_ns - 1) / (individuals_block_ns), NODES*individuals_block_ns >> >(d_population); //TODO revisar numeros
			//next_state_v2 << <POPULATION,INDIVIDUAL_SIZE >> >(d_population); //TODO revisar numeros
			//gpuErrchk(cudaPeekAtLastError());
			//population_fitness<<<POPULATION,1>>>(d_population,POPULATION, d_current_fitness, i);
			/** POPULATION FITNESS **/

			sequence << <(POPULATION + 1023) / 1024, 1024 >> >(d_indexes);
			gpuErrchk(cudaPeekAtLastError());

			compute_fitness(d_population, link_distances, d_x_links, link_fitness, handle, node_distances, d_x_nodes, node_fitness, d_current_fitness, d_goal_links, d_init_nodes, d_goal_nodes);

		//	min << <1, 1 >> >(d_population, d_current_fitness, POPULATION, i, d_BEST_INDIVIDUAL_FITNESS, d_BEST_INDIVIDUAL);
		min_v2(d_population, device_ptr_fitness, min_ptr, i, f, &h_BEST_INDIVIDUAL_FITNESS_HOST, &h_BEST_INDIVIDUAL);
			gpuErrchk(cudaPeekAtLastError());

			elite_selection(d_population, link_distances, d_x_links, link_fitness, handle, node_distances, d_x_nodes, node_fitness, d_current_fitness, device_ptr_fitness, d_indexes, dev_indexes, d_indexes_best);
			//roulette_sel(d_population, link_distances, d_x_links, link_fitness, handle, node_distances, d_x_nodes, node_fitness, d_current_fitness, device_ptr_fitness, d_indexes, dev_indexes, d_indexes_best, devStates, dev_total_fitness);

			gpuErrchk(cudaPeekAtLastError());
			/*
			HANDLE_ERROR(
				cudaMemcpy(&h_BEST_INDIVIDUAL_FITNESS_HOST, d_BEST_INDIVIDUAL_FITNESS, sizeof(float), cudaMemcpyDeviceToHost)
				);
			HANDLE_ERROR(
				cudaMemcpy(&h_BEST_INDIVIDUAL, d_BEST_INDIVIDUAL, sizeof(network), cudaMemcpyDeviceToHost)
				);
			fprintf(f, "%d,%.8f\n", i, h_BEST_INDIVIDUAL_FITNESS_HOST);

			*/
			print_progress((float)i / (float)GENERATIONS, h_BEST_INDIVIDUAL_FITNESS_HOST);
			gpuErrchk(cudaPeekAtLastError());

			generateLinkCrossoverIndexes << < (ELITE_MEMBERS / 2 + 447) / 448, 448 >> >(link_crossover_indexes, devStates);
			gpuErrchk(cudaPeekAtLastError());
			//generateRuleCrossoverIndexes << <(ELITE_MEMBERS / 2 + 31) / 32, 32 >> >(rule_crossover_indexes, devStates);
			//gpuErrchk(cudaPeekAtLastError());

			crossover<512> << <ELITE_MEMBERS / 2, (NODES*MAX_INPUTS) / 2 >> >(d_population, d_indexes_best, link_crossover_indexes);
			gpuErrchk(cudaPeekAtLastError());

			mutation << <(POPULATION + 32 * 5 + 1) / (32 * 5), 32 * 5 >> >(d_population, NODES, MAX_INPUTS, RULES_PER_NODE, LINK_MUTATION_PROB, RULE_MUTATION_PROB, devStates);
			//	mutation << <(POPULATION + (32 * 20) - 1) / (32 * 20), 32 * 20 >> >(d_population, NODES, MAX_INPUTS, RULES_PER_NODE, LINK_MUTATION_PROB, RULE_MUTATION_PROB, devStates);
			gpuErrchk(cudaPeekAtLastError());

			migrate(i, MIGRATION_FREQUENCY, d_population, d_indexes_best);

			//	gpuErrchk( cudaDeviceSynchronize() );
		}
		// Retrieve data from device

		HANDLE_ERROR(cudaEventRecord(stop));
		HANDLE_ERROR(cudaEventSynchronize(stop));
		HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
		fprintf(f, "\nTime to generate:  %3.1f ms \n", time);
		print_network_file(h_BEST_INDIVIDUAL, f);

		fclose(f);

		cudaFree(d_population);
		cudaFree(total_fitness);
		cudaFree(d_BEST_INDIVIDUAL);
		cudaFree(d_BEST_INDIVIDUAL_FITNESS);
		cudaFree(d_indexes);
		cudaFree(d_current_fitness);
		cudaFree(link_fitness);
		cudaFree(node_fitness);
		cudaFree(d_x_links);
		cudaFree(d_x_nodes);
		cudaFree(d_indexes_best);
		cudaFree(node_distances);
		cudaFree(link_distances);
		cudaFree(devStates);
		cublasDestroy(handle);
	}
	cudaDeviceReset();
	return 0;
}