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

/* Do NOT modify this constants */

#define ROULETTE 0
#define ELITE 1
#define CELLULAR 2

#define MOST 0
#define ABSOLUTE_REPRESSOR 1
#define JOINT_ACTIVATORS 2
#define JOINT_REPRESSORS 3

#define MAX_INPUTS 32
#define RULES_PER_NODE 1
#define NODES 32

#define TOTAL_LINKS (NODES*MAX_INPUTS)

/********************************/
/* Modify this constants depending on your needs */
#define POPULATIONS 2	//DO NOT FORGET TO FILL LAMDA_HOST PROPERLY

#define POPULATION (1024*POPULATIONS)

#define COLS 32		// Square root of a single population size
#define ELITE_MEMBERS (POPULATION)
#define ELEMENTS_TO_MIGRATE (16*POPULATIONS)
#define MIGRATION_FREQUENCY 16

#define MAX_CONNECTIVITY_DISTANCE 0.1

#define GENERATIONS 100000
#define EXECUTIONS 4
#define LINK_MUTATION_PROB 0.001  
#define RULE_MUTATION_PROB 0.001
#define SELECTION CELLULAR

typedef struct
{
	char links[TOTAL_LINKS];
	char rules[RULES_PER_NODE*NODES];
} network;

const unsigned char H_INIT_NODES[NODES] = { 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1 };
const unsigned char H_GOAL_NODES[NODES] = { 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

network population[POPULATION];

__device__ const network INIT_NETWORK_DEVICE = {
	{ 1, 0, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }
};

network INIT_NETWORK_HOST = {
	{ 1, 0, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }
};
network GOAL_NETWORK_HOST = {
	{ 1, 0, 0, 1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }
};

float LAMBDA_HOST[POPULATIONS] = { .9f, .9f };

__constant__ float LAMBDA_VALUES[POPULATIONS];

__constant__ float INIT_CONNECTIVITY;

float BEST_FITNESS_HOST = 1.0f;


/// <summary>
/// Simple hash function
/// </summary>
/// <param name="a">Number to hash</param>
/// <returns>Hashed number</returns>
__device__ unsigned int WangHash(unsigned int a){
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}


/// <summary>
/// Generates a random float number between [0,1)
/// </summary>
/// <param name="globalState">curandState of the thread.</param>
/// <returns>Random float</returns>
__device__ float generate(curandState* globalState)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;

	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}


/// <summary>
/// Generates 'count' random float numbers between [0,1)
/// </summary>
/// <param name="globalState">curandState of the thread.</param>
/// <param name="values">Generated values.</param>
/// <param name="count">Number of values to generate.</param>
/// <returns></returns>
__device__ void generate_v2(curandState* globalState, float * values, unsigned int count)
{
	const unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;

	curandState localState = globalState[ind];
	for (int i = 0; i < count; i++){
		values[i] = curand_uniform(&localState);
	}

	globalState[ind] = localState;

}

/// <summary>
/// Initializes cuRAND states with different sequence numbers per thread. If many states are created, it may be very slow.
/// </summary>
/// <param name="state">Pointer to allocated curandState.</param>
/// <returns></returns>
__global__ void setup_kernel(curandState * state)
{
	const unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int seed = 1234;
	curand_init(seed, id, 0, &state[id]);
}

/// <summary>
/// Initializes cuRAND states with seed per thread. Much faster than setup_kernel.
/// </summary>
/// <param name="state">Pointer to allocated curandState.</param>
/// <returns></returns>
__global__ void setupCurandDiffSeed(curandState * state)
{
	const unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int seed = (unsigned int)clock64();
	curand_init(WangHash(seed) + id, 0, 0, &state[id]);
}


/// <summary>
/// Shuffles the specified array.
/// </summary>
/// <param name="array">The array.</param>
/// <param name="n">Size</param>
/// <param name="globalState">curandState of the thread.</param>
/// <returns></returns>
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


/// <summary>
/// Prints the specified char array.
/// </summary>
/// <param name="array">The array.</param>
/// <param name="size">Size.</param>
/// <returns></returns>
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

/// <summary>
/// Prints the specified float array.
/// </summary>
/// <param name="array">The array.</param>
/// <param name="size">Size</param>
/// <returns></returns>
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

/// <summary>
/// Prints the specified char array into a file.
/// </summary>
/// <param name="array">The array.</param>
/// <param name="size">Size.</param>
/// <param name="f">Output file.</param>
/// <returns></returns>
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

/// <summary>
/// Prints both links and rules of a given chromosome.
/// </summary>
/// <param name="individual">The individual.</param>
/// <returns></returns>
__device__ __host__ void print_network(network individual)
{
	printf("--\n");
	print_array(individual.links, TOTAL_LINKS);
	print_array(individual.rules, RULES_PER_NODE * NODES);
}

/// <summary>
/// Prints both links and rules of a given chromosome into a file.
/// </summary>
/// <param name="individual">The individual.</param>
/// <param name="f">Output file.</param>
/// <returns></returns>
__host__ void print_network_file(network individual, FILE *f)
{
	fprintf(f, "\nLINKS: ");
	print_array_file(individual.links, TOTAL_LINKS, f);
	fprintf(f, "\nRULES: ");
	print_array_file(individual.rules, RULES_PER_NODE * NODES, f);
}

/// <summary>
/// Prints an entire population.
/// </summary>
/// <param name="population">The population.</param>
/// <param name="population_size">Number of individuals.</param>
/// <returns></returns>
__device__ __host__ void print_population(network population[], int population_size)
{
	int i;
	printf("------------------- POPULATION -------------------\n");
	for (i = 0; i < population_size; ++i)
	{
		print_network(population[i]);
	}
	printf("---------------------------------------------------\n");
}

/// <summary>
/// Shows the algorithm's progress.
/// </summary>
/// <param name="progress">The progress.</param>
/// <param name="best">Fitness of the best individual.</param>
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

		printf("] %.2f -- Best: %.8f\r", progress * 100, best);

		fflush(stdout);
	}
}

/// <summary>
/// Prints the algorithm's current parameters.
/// </summary>
void printParameters(){
	unsigned int width = 35;
	printf("TOTAL POPULATION: %*d\n", width, POPULATION);
	if (POPULATIONS > 1){
		printf("ISLANDS (POPULATIONS): %*d\n", width, POPULATIONS);
		printf("MIGRATIONS/ISLAND: %*d\n", width, ELEMENTS_TO_MIGRATE / POPULATIONS);
		printf("MIGRATION FREQUENCY: %*d\n", width, MIGRATION_FREQUENCY);
	}
	printf("LINK MUTATION PROB: %*.3f\n", width, LINK_MUTATION_PROB);
	printf("RULE MUTATION PROB: %*.3f\n", width, RULE_MUTATION_PROB);
	printf("GENERATIONS: %*d\n", width, GENERATIONS);
}


/// <summary>
/// Applies the given rule to a node until a stable state is reached.
/// </summary>
/// <param name="links">Node inputs.</param>
/// <param name="rule">The rule.</param>
/// <param name="nodes">At least, all the nodes of an individual.</param>
/// <param name="node_index">Node index into nodes[].</param>
/// <returns></returns>
template <unsigned int individualsPerBlock>
__device__ void applyRules(const char links[], const char rule, char nodes[], const unsigned int node_index)
{

	/*
	Three positions per thread.
	[0] = activator
	[1] = repressor
	[2] = null/invalid link
	*/
	__shared__ char sharedInputCount[3 * individualsPerBlock*NODES];

	/*
	Considers only links coming from active nodes.
	*/
	char validLinkValue;
	/*
	Index to increase. Offset must be added.
	0 = activator
	1 = repressor
	2 = null/invalid link
	*/
	char inputCountIndex;

	/*
	New possible states depending on the input count result.
	[0] = node deactivated
	[1] = node activated
	[2] = not changed
	*/
	char states[3] = { 0, 1, nodes[node_index] };
	/*
	Index within states[] if [rule] is applied.
	states[MOST] = New state if rule of the most is applied.
	states[ABSOLUTE_REPRESSOR] = New state if absolute repressor rule is applied.
	states[JOINT_ACTIVATORS] = New state if joint activators rule is applied.
	states[JOINT_REPRESSORS] = New state if joint repressors rule is applied.

	This double-pointer approach is used because a node may preserve its state.
	*/
	char ruleIndexedStates[4];

	const char stableState = 5;
	char indexToMostNewState, indexToAbsReprNewState, indexToJointActNewState, indexToJointReprNewState;
	for (char j = 0; j < stableState; j++){

		// Reset counters
		sharedInputCount[0 + node_index * 3] = 0;
		sharedInputCount[1 + node_index * 3] = 0;
		sharedInputCount[2 + node_index * 3] = 0;

		// Inputs count
		for (char i = 0; i < MAX_INPUTS; i++)
		{
			validLinkValue = (links[i] * nodes[i + (node_index >> 5)*NODES]);

			inputCountIndex = (-(validLinkValue - 1)*(3 * validLinkValue + 4)) >> 1;

			sharedInputCount[inputCountIndex + node_index * 3]++;
		}

		// Generate index to all posible states
		indexToMostNewState = (sharedInputCount[0 + node_index * 3] > sharedInputCount[1 + node_index * 3]) - (sharedInputCount[0 + node_index * 3] < sharedInputCount[1 + node_index * 3]); // (+ > -) - ( + < -)
		ruleIndexedStates[MOST] = (-(indexToMostNewState + 1)*(3 * indexToMostNewState - 4)) >> 1;

		indexToAbsReprNewState = sharedInputCount[1 + node_index * 3] > 0;
		ruleIndexedStates[ABSOLUTE_REPRESSOR] = 2 - (indexToAbsReprNewState << 1);

		indexToJointActNewState = sharedInputCount[0 + node_index * 3] > 1;
		ruleIndexedStates[JOINT_ACTIVATORS] = 2 - indexToJointActNewState;

		indexToJointReprNewState = sharedInputCount[1 + node_index * 3] > 1;
		ruleIndexedStates[JOINT_REPRESSORS] = 2 - (indexToJointReprNewState << 1);

		// Change node state
		nodes[node_index] = states[2] = states[ruleIndexedStates[rule]];

		//	__syncthreads(); // Not needed but increases performance. Due to bank conflicts?
	}

}

/// <summary>
/// Calculates normalized single link distance and saves it into an column-major ordered array.
/// </summary>
/// <param name="population">The population.</param>
/// <param name="distances">Column-major ordered array to hold the distances.</param>
/// <param name="goal_links">The goal links.</param>
/// <returns></returns>
template <unsigned int links_per_node>
__global__ void linkDistance(network population[], float *distances, char goal_links[]){

	const unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	const unsigned int individual = (index*links_per_node) / TOTAL_LINKS;
	const unsigned int link = (index*links_per_node) % TOTAL_LINKS;
	const unsigned int populationIndex = individual / (POPULATION / POPULATIONS);

	char links[links_per_node];
	char goal[links_per_node];

	if (individual < POPULATION){

		// Vectorized memory load to increase performance
		int2 *p = reinterpret_cast<int2*>(population[individual].links + link);
		int2 links_vec = p[0];
		reinterpret_cast<int2*>(links)[0] = links_vec;

		int2 *p2 = reinterpret_cast<int2*>(goal_links + link);
		int2 goal_vec = p2[0];
		reinterpret_cast<int2*>(goal)[0] = goal_vec;

		for (char i = 0; i < links_per_node; i++){
			float distance = (links[i] != goal[i]);
			distance *= (1 - LAMBDA_VALUES[populationIndex]) / (TOTAL_LINKS);
			distances[IDX2C(individual, link + i, POPULATION)] = distance;
		}

	}
}

/// <summary>
/// Calculates normalized single node distance and saves it into an column-major ordered array.
/// </summary>
/// <param name="population">The population.</param>
/// <param name="distances">Column-major ordered array to hold the distances.</param>
/// <param name="init_nodes">The initial nodes.</param>
/// <param name="goal_nodes">The goal nodes.</param>
/// <returns></returns>
template <unsigned int individualsPerBlock>
__global__ void nodeDistance(network population[], float *distances, char init_nodes[], char goal_nodes[]){

	const unsigned char node = threadIdx.x % NODES;

	__shared__ char nodes[individualsPerBlock*NODES];

	/*
	Offset within the total population.
	*/
	__shared__ unsigned int offset;

	char nodeInputs[MAX_INPUTS];

	if (threadIdx.x == 0){
		offset = blockIdx.x * individualsPerBlock;
	}

	__syncthreads();

	// Node state is not saved between generations, so it's retrieved from the initial network
	nodes[threadIdx.x] = init_nodes[node];

	const unsigned int individual = (threadIdx.x / NODES) + offset;
	const unsigned int populationIndex = individual / (POPULATION / POPULATIONS);

	if (individual < POPULATION){

		const unsigned char rule = population[individual].rules[node];

		//TO-DO En la primera iteracion puedo calcular directamente represores y activadores
		// Retrieve all 32 inputs in two 16-char read loads
		for (int i = 0; i < 2; i++){
			int4 *p = reinterpret_cast<int4*>(population[individual].links + node*MAX_INPUTS + i * 16);
			int4 links_vec = p[0];
			reinterpret_cast<int4*>(nodeInputs)[i] = links_vec;
		}

		// Apply rules in shared memory, since node state is not saved
		applyRules<individualsPerBlock>(nodeInputs, rule, nodes, threadIdx.x);

		float distance = (nodes[threadIdx.x] != goal_nodes[node]);
		distance *= LAMBDA_VALUES[populationIndex] / NODES;
		distances[IDX2C(individual, node, POPULATION)] = distance;
	}
}

/// <summary>
/// Sums all the single link distances, so we obtain the total link fitness per individual.
/// </summary>
/// <param name="singleLinkDistances">Single link distances matrix.</param>
/// <param name="onesVector">Vector of TOTAL_LINKS size filled with ones</param>
/// <param name="linkFitnessPerIndividual">Vector to hold the link fitness per individual.</param>
/// <param name="row">Number of rows of singleLinkDistances, that is, INVIDIDUAL_SIZE</param>
/// <param name="col">Number of columns of singleLinkDistances, that is, POPULATION.</param>
/// <param name="handle">cublasHandle</param>
/// <returns>Error status</returns>
cublasStatus_t sumSingleLinkDistances(const float* singleLinkDistances, const float* onesVector, float* linkFitnessPerIndividual, const int row, const int col, cublasHandle_t handle){

	// Level 2 calculation y = alpha * A * x + beta * y
	float alf = 1.f;
	float beta = 0.f;

	return cublasSgemv(handle, CUBLAS_OP_N, col, row, &alf, singleLinkDistances, col, onesVector, 1, &beta, linkFitnessPerIndividual, 1);//swap col and row
}

/// <summary>
/// Sums all the single node distances, so we obtain the total node fitness per individual.
/// </summary>
/// <param name="singleNodeDistances">Single node distances matrix.</param>
/// <param name="onesVector">Vector of NODES size filled with ones.</param>
/// <param name="nodeFitnessPerIndividual">Vector to hold the node fitness per individual.</param>
/// <param name="row">Number of rows of singleNodeDistances, that is, NODES</param>
/// <param name="col">Number of columns of singleNodeDistances, that is, POPULATION.</param>
/// <param name="handle">cublasHandle</param>
/// <returns>Error status</returns>
cublasStatus_t sumSingleNodeDistances(const float* singleNodeDistances, const float* onesVector, float* nodeFitnessPerIndividual, const int row, const int col, cublasHandle_t handle){

	// level 2 calculation y = alpha * A * x + beta * y
	float alf = 1.f;
	float beta = 0.f;

	return cublasSgemv(handle, CUBLAS_OP_N, col, row, &alf, singleNodeDistances, col, onesVector, 1, &beta, nodeFitnessPerIndividual, 1);//swap col and row

}

/// <summary>
/// Sums link and node fitness obtaining fitness per individual.
/// </summary>
/// <param name="linkFitness">Link fitness per individual.</param>
/// <param name="nodeFitness">Node fitness per individual.</param>
/// <param name="populationFitness">Total fitness per individual.</param>
/// <param name="handle">cublasHandle</param>
/// <returns>Error status</returns>
cublasStatus_t populationFitness(const float* linkFitness, const float* nodeFitness, float* populationFitness, cublasHandle_t handle){

	float alf = 1.f;
	float beta = 1.f;
	return cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, POPULATION, 1, &alf, linkFitness, POPULATION, &beta, nodeFitness, POPULATION, populationFitness, POPULATION);

}

/// <summary>
/// Generates random rules per individual.
/// </summary>
/// <param name="individualRules">The individual rules.</param>
/// <param name="size">Number of rules to generate.</param>
/// <param name="globalState">curandState.</param>
/// <returns></returns>
__device__ void generateRandomRules(char individualRules[], const int size, curandState * globalState)
{
	int i, rule;
	float randoms[RULES_PER_NODE*NODES];
	generate_v2(globalState, randoms, RULES_PER_NODE*NODES);
	for (i = 0; i < size; i++)
	{
		rule = randoms[i] * 4;	// Four possible rules
		individualRules[i] = rule;
	}
}


/// <summary>
/// Generates the initial population.
/// </summary>
/// <param name="population">The population.</param>
/// <param name="globalState">curandState</param>
/// <returns></returns>
__global__ void generateInitialPopulation(network population[], curandState* globalState)
{

	const unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;

	network individual;

	if (index < POPULATION){
		// Copy and shuffle initial links
		individual = INIT_NETWORK_DEVICE;
		shuffle(individual.links, MAX_INPUTS * NODES, globalState);
		// Generate random rules
		generateRandomRules(individual.rules, RULES_PER_NODE * NODES, globalState);
		// Save to device memory
		population[index] = individual;
	}

}

/// <summary>
/// Mutates every individual of the population within a probability.
/// </summary>
/// <param name="population">The population.</param>
/// <param name="globalState">curandState</param>
/// <returns></returns>
__global__ void mutation(network *population, curandState *globalState)
{

	const unsigned int individual_index = (blockIdx.x*blockDim.x + threadIdx.x);

	const unsigned char rules[] = { 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2 };
	const char links[] = { 0, 1, 1, -1, 0, -1 };
	/*
	Old values that will mutate into new ones.
	*/
	char oldLinkValue, oldRuleValue;
	/*
	Random indexes.
	*/
	unsigned int individualLinkIndex, individualRuleIndex;
	/*
	Indexes for the local links and rules arrays.
	*/
	unsigned char localLinkIndex, localRuleIndex;

	float randoms[6];
	generate_v2(globalState, randoms, 6);

	if (individual_index < POPULATION){

		if (randoms[0] <= LINK_MUTATION_PROB)
		{
			// Random link position
			individualLinkIndex = randoms[1] * (TOTAL_LINKS);
			// Previous link value
			oldLinkValue = population[individual_index].links[individualLinkIndex];
			// Index of the new value within the links array
			// Index returned will point a different value from old one. E.g. oldLink = 1, localLinkIndex = [4,5] -> new value = [0,-1]
			localLinkIndex = randoms[2] * 2 + (oldLinkValue + 1) * 2;
			// Save the changes
			population[individual_index].links[individualLinkIndex] = links[localLinkIndex];
		}

		if (randoms[3] <= RULE_MUTATION_PROB)
		{
			// Random rule position
			individualRuleIndex = randoms[4] * RULES_PER_NODE*NODES;
			// Previous rule value
			oldRuleValue = population[individual_index].rules[individualRuleIndex];
			// Index of the new value within the rules array
			// Index returned will point a different value from old one. E.g. oldRule = 1, localRuleIndex = [3,4,5] -> new value = [0,2,3]
			localRuleIndex = randoms[5] * 3 + oldRuleValue * 3;
			// Save the changes
			population[individual_index].rules[individualRuleIndex] = rules[localRuleIndex];
		}

	}

}


template <unsigned int blockSize>
/// <summary>
/// Calcules the connectivity of a given individual using a reduction.
/// </summary>
/// <param name="links">The links.</param>
/// <param name="connectivity">Connectivity result.</param>
/// <returns></returns>
__device__ void calculeConnectivity(const char links[], float *connectivity){

	const unsigned int tid = threadIdx.x;

	__shared__ char sdata[TOTAL_LINKS / 2];

	/*
	Sum of existing links.
	*/
	unsigned int mySum;

	if (tid < TOTAL_LINKS){
		mySum = links[tid] != 0;
	}

	if (tid + blockSize < TOTAL_LINKS){
		mySum += (links[tid + blockSize] != 0);
	}

	sdata[tid] = mySum;

	__syncthreads();

	// Do reduction in shared mem
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
	// Fully unroll reduction within a single warp
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

	// Write value
	if (tid == 0){
		*connectivity = ((float)mySum) / TOTAL_LINKS;
	}
	__syncthreads();
}

/// <summary>
/// Fills indices array with the index of the best individual within a neighborhood.
/// </summary>
/// <param name="populationFitness">The population fitness.</param>
/// <param name="selectedIndices">The selected indices.</param>
/// <returns></returns>
template <unsigned int blockSize, unsigned int populationOffset, int cols>
__global__ void cellularNeighborhood(const float populationFitness[], int selectedIndices[]){

	__shared__ float fitnessValues[blockSize];

	const int tid = threadIdx.x;

	const unsigned int populationIndex = blockIdx.x;		// One block per population/neighborhood
	const unsigned int row = tid / cols;

	const int up = (tid - cols + POPULATION / POPULATIONS) % (POPULATION / POPULATIONS);
	const int down = (tid + cols) % POPULATION / POPULATIONS;
	const int left = (((tid - 1) % cols + cols) % cols) + row*cols;
	const int right = (tid + 1) % cols + row*cols;

	fitnessValues[threadIdx.x] = populationFitness[tid + populationOffset * populationIndex];

	__syncthreads();

	unsigned int indexBest = up;
	float bestFitness = fitnessValues[up];

	if (fitnessValues[down] < bestFitness){
		indexBest = down;
		bestFitness = fitnessValues[down];
	}
	if (fitnessValues[right] < bestFitness){
		indexBest = right;
		bestFitness = fitnessValues[right];
	}
	if (fitnessValues[left] < bestFitness){
		indexBest = left;
	}

	selectedIndices[tid + populationOffset * populationIndex] = indexBest;

}
/// <summary>
/// Updates the best individual and writes the current best fitness into a file.
/// </summary>
/// <param name="population">The population.</param>
/// <param name="populationFitness">The population fitness.</param>
/// <param name="generation">Current generation</param>
/// <param name="f">Output file.</param>
/// <param name="bestFitness">The best fitness.</param>
/// <param name="bestIndividual">The best individual.</param>
/// <param name="handle">cuBLAS handle</param>
void updateBestIndividual(network population[], const float populationFitness[], const int generation, FILE *f, float * bestFitness, network * bestIndividual, cublasHandle_t handle){

	int position = 0;
	float min = 1.f;

	cublasStatus_t error = cublasIsamin(handle, POPULATION, populationFitness, 1, &position);

	position -= 1;		// CUBLAS uses 1-base indexing
	if (error == CUBLAS_STATUS_SUCCESS && position >= 0 && position < POPULATION){
		HANDLE_ERROR(
			cudaMemcpy(&min, populationFitness + position, sizeof(float), cudaMemcpyDeviceToHost)
			);


		// Update if current fitness is better than the previous one
		if (min < *(bestFitness)){
			*bestFitness = min;
			HANDLE_ERROR(
				cudaMemcpy(bestIndividual, population + position, sizeof(network), cudaMemcpyDeviceToHost)
				);
		}
		// Write to file
		fprintf(f, "%d,%.8f\n", generation, *bestFitness);
	}

}

/// <summary>
/// Sorts an indices array according to the population(s) fitness, so the resulting order will be accord their fitness (best-first order).
/// </summary>
/// <param name="populationFitness">The population fitness.</param>
/// <param name="indices">The indices.</param>
void sort_population(thrust::device_ptr<float> populationFitness, thrust::device_ptr<int> indices){

	for (int i = 0; i < (POPULATIONS); i++){
		thrust::sort_by_key(populationFitness + i*POPULATION / POPULATIONS, populationFitness + (i + 1)*POPULATION / POPULATIONS, indices + i*POPULATION / POPULATIONS);
	}

}

/// <summary>
/// Resets the indices array. If more than one population exists, each will start from zero.
/// </summary>
/// <param name="indices">The indices.</param>
/// <returns></returns>
__global__ void sequence(int * indices){
	const unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	if (index < POPULATION){
		indices[index] = index % (POPULATION / POPULATIONS);
	}
}

/// <summary>
/// Generates random link crossover points.
/// </summary>
/// <param name="points">The points.</param>
/// <param name="globalState">curandState.</param>
/// <returns></returns>
__global__ void generateLinkCrossPoints(int *points, curandState *globalState){
	const unsigned int index = threadIdx.x + blockDim.x *blockIdx.x;
	const int N = ELITE_MEMBERS / 2;
	if (index < N){
		points[index] = generate(globalState)*TOTAL_LINKS;
	}
}
/// <summary>
/// Generates the rule crossover points.
/// </summary>
/// <param name="points">The points.</param>
/// <param name="globalState">curandState.</param>
/// <returns></returns>
__global__ void generateRuleCrossoverPoints(int *points, curandState *globalState){
	const unsigned int index = threadIdx.x + blockDim.x *blockIdx.x;
	const int N = ELITE_MEMBERS / 2;
	if (index < N){
		points[index] = generate(globalState)*RULES_PER_NODE*NODES;
	}
}


/// <summary>
/// Crossovers the individuals pointed by the given indexes. One crossover per block.
/// </summary>
/// <param name="population">The population.</param>
/// <param name="selectedIndices">The selected individual indices.</param>
/// <param name="linkCrossIndices">The crossovers point.</param>
/// <returns></returns>
template <unsigned int blockSize, unsigned int offset, unsigned int selection>
__global__ void crossover(network population[], const int *selectedIndices, const int *linkCrossIndices){

	const unsigned int populationIndex = (blockIdx.x * 2) / (POPULATION / POPULATIONS);

	int indexParent1;
	int indexParent2;

	if (selection == CELLULAR){
		indexParent1 = blockIdx.x;
		indexParent2 = selectedIndices[blockIdx.x] + offset*populationIndex;
	}
	else{
		indexParent1 = selectedIndices[blockIdx.x * 2] + offset*populationIndex;
		indexParent2 = selectedIndices[blockIdx.x * 2 + 1] + offset*populationIndex;
	}

	const unsigned int tid = threadIdx.x;

	__shared__ char linksChild1[TOTAL_LINKS];
	__shared__ char linksChild2[TOTAL_LINKS];

	__shared__ unsigned char rulesChild2[RULES_PER_NODE*NODES];

	const int linkCrossPoint = linkCrossIndices[blockIdx.x];
	const char ruleCrossPoint = (linkCrossPoint / NODES) + 1;

	/** Copy the children's links **/

	if (tid < TOTAL_LINKS){
		if (tid < linkCrossPoint){
			linksChild1[tid] = population[indexParent1].links[tid];
			linksChild2[tid] = population[indexParent2].links[tid];
		}
		else{
			linksChild1[tid] = population[indexParent2].links[tid];
			linksChild2[tid] = population[indexParent1].links[tid];
		}

	}
	if (tid + blockSize < TOTAL_LINKS){
		if (tid + blockSize < linkCrossPoint){
			linksChild1[tid + blockSize] = population[indexParent1].links[tid + blockSize];
			linksChild2[tid + blockSize] = population[indexParent2].links[tid + blockSize];
		}
		else{
			linksChild1[tid + blockSize] = population[indexParent2].links[tid + blockSize];
			linksChild2[tid + blockSize] = population[indexParent1].links[tid + blockSize];
		}
	}

	if (tid < RULES_PER_NODE*NODES && tid >= ruleCrossPoint){
		rulesChild2[tid] = population[indexParent1].rules[tid];
	}


	__syncthreads();

	// Calcule connectivity of each child and replace parents if pre-fitness is satisfied

	__shared__ float child1Connectivity, child2Connectivity;

	/** Child 1 **/
	calculeConnectivity<blockSize>(linksChild1, &child1Connectivity);

	if (fabsf(child1Connectivity - INIT_CONNECTIVITY) < MAX_CONNECTIVITY_DISTANCE){
		// Links
		if (tid > linkCrossPoint && tid < TOTAL_LINKS){
			population[indexParent1].links[tid] = linksChild1[tid];
		}
		if (blockSize >= 512 && ((tid + 512) > linkCrossPoint) && (tid + 512) < TOTAL_LINKS){
			population[indexParent1].links[tid + 512] = linksChild1[tid + 512];
		}

		// Rules
		if (tid < RULES_PER_NODE*NODES && tid >= ruleCrossPoint){
			population[indexParent1].rules[tid] = population[indexParent2].rules[tid];
		}
	}

	/** Child 2 **/
	calculeConnectivity<blockSize>(linksChild2, &child2Connectivity);

	if (fabsf(child2Connectivity - INIT_CONNECTIVITY) < MAX_CONNECTIVITY_DISTANCE){
		// Links
		if (tid > linkCrossPoint && tid < TOTAL_LINKS){
			population[indexParent2].links[tid] = linksChild2[tid];
		}
		if (blockSize >= 512 && ((tid + 512) > linkCrossPoint) && (tid + 512) < TOTAL_LINKS){
			population[indexParent2].links[tid + 512] = linksChild2[tid + 512];
		}

		// Rules
		if (tid < RULES_PER_NODE*NODES && tid >= ruleCrossPoint){
			population[indexParent2].rules[tid] = rulesChild2[tid];
		}
	}

}

/// <summary>
/// Calculates the total fitness of each population. A inverse operator is applied so higher values will correspond to better fitness.
/// </summary>
/// <param name="populationFitness">The population(s) fitness.</param>
/// <param name="fitnessSum">The fitness sum.</param>
void total_fitness(thrust::device_ptr<float> populationFitness, thrust::device_ptr<float> fitnessSum){

	inverse<float> unary_op;
	thrust::plus<float> binary_op;
	float init = 0;

	for (int i = 0; i < (POPULATIONS); i++){
		fitnessSum[i] = thrust::transform_reduce(populationFitness + i*POPULATION / POPULATIONS, populationFitness + (i + 1)*POPULATION / POPULATIONS, unary_op, init, binary_op);
	}

}

/// <summary>
/// Performs a roulette selection among the individuals.
/// </summary>
/// <param name="populationFitness">The population fitness.</param>
/// <param name="chosenIndices">The chosen indices.</param>
/// <param name="fitnessSum">The fitness sum of each population.</param>
/// <param name="globalState">curandState.</param>
/// <returns></returns>
template <unsigned int populationOffset>
__global__ void rouletteSelection(float populationFitness[], int chosenIndices[], thrust::device_ptr<float> fitnessSum, curandState *globalState){

	const unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	const unsigned int populationIndex = index / (POPULATION / POPULATIONS);

	if (index < POPULATION){
		float random = generate(
			globalState) * fitnessSum[populationIndex];
		int pick = populationOffset*populationIndex;
		double offset = 0;
		const float individualFitness = 1 / populationFitness[pick];
		while (random > offset){
			offset += individualFitness;
			pick++;
		}
		chosenIndices[index] = pick % (POPULATION / POPULATIONS);
	}

}

/// <summary>
/// Migrates the number of individuals specified by ELEMENTS_TO_MIGRATE constant between adjacents populations.
/// </summary>
/// <param name="population">The entire population.</param>
/// <param name="orderedIndices">The ordered indices.</param>
/// <returns></returns>
template <unsigned int elementsPerPopulation, unsigned int offset>
__global__ void migrate(network population[], int orderedIndices[]){
	const unsigned int index = (blockIdx.x * 2) + (threadIdx.x >> 6);
	const unsigned int tid = threadIdx.x % 64;
	const unsigned int populationIndex = index / elementsPerPopulation;
	const unsigned int offsetElement = (POPULATION / POPULATIONS) * 2 - elementsPerPopulation;

	if (index < ELEMENTS_TO_MIGRATE){
		const unsigned int idBest = populationIndex*offset + index%elementsPerPopulation;
		const unsigned int idWorst = (idBest + offsetElement) % ELITE_MEMBERS;
		const unsigned int bestIndividual = orderedIndices[idBest];
		const unsigned int worstIndividual = orderedIndices[idWorst];
		int4 *pTarget, *pSource;

		pTarget = reinterpret_cast<int4*>(population[worstIndividual + ((populationIndex + 1)*POPULATION / POPULATIONS) % POPULATION].links + tid * 16);
		pSource = reinterpret_cast<int4*>(population[(populationIndex*POPULATION / POPULATIONS) + bestIndividual].links + tid * 16);

		*pTarget = *pSource;

		if (tid < NODES){
			population[worstIndividual + ((populationIndex + 1)*POPULATION / POPULATIONS) % POPULATION].rules[tid] = population[(populationIndex*POPULATION / POPULATIONS) + bestIndividual].rules[tid];
		}
	}

}

/// <summary>
/// Computes fitness for every individual of the population(s).
/// </summary>
/// <param name="stream1">The stream1.</param>
/// <param name="stream2">The stream2.</param>
/// <param name="population">The population.</param>
/// <param name="singleLinkDistances">The single link distances.</param>
/// <param name="onesLinksVector">The ones links vector.</param>
/// <param name="linkFitnessPerIndividual">The link fitness per individual.</param>
/// <param name="handle">cublasHandle.</param>
/// <param name="singleNodeDistances">The single node distances.</param>
/// <param name="onesNodesVector">The ones nodes vector.</param>
/// <param name="nodeFitnessPerIndividual">The node fitness per individual.</param>
/// <param name="popFitness">The population fitness.</param>
/// <param name="goalLinks">The goal links.</param>
/// <param name="initNodes">The initial nodes.</param>
/// <param name="goalNodes">The goal nodes.</param>
void computeFitness(cudaStream_t stream1, cudaStream_t stream2, network * population, float * singleLinkDistances, float * onesLinksVector, float * linkFitnessPerIndividual, cublasHandle_t handle, float * singleNodeDistances, float * onesNodesVector, float * nodeFitnessPerIndividual, float * popFitness, char goalLinks[], char initNodes[], char goalNodes[]){
	linkDistance<8> << <(POPULATION*TOTAL_LINKS / 8 + 32 * 4 - 1) / (32 * 4), 32 * 4 >> >(population, singleLinkDistances, goalLinks);
	gpuErrchk(cudaPeekAtLastError());

	nodeDistance <4> << <(POPULATION*NODES + 4 * 32 - 1) / (32 * 4), 32 * 4 >> >(population, singleNodeDistances, initNodes, goalNodes);
	gpuErrchk(cudaPeekAtLastError());

	sumSingleLinkDistances(singleLinkDistances, onesLinksVector, linkFitnessPerIndividual, TOTAL_LINKS, POPULATION, handle);
	gpuErrchk(cudaPeekAtLastError());

	sumSingleNodeDistances(singleNodeDistances, onesNodesVector, nodeFitnessPerIndividual, NODES, POPULATION, handle);
	gpuErrchk(cudaPeekAtLastError());

	populationFitness(linkFitnessPerIndividual, nodeFitnessPerIndividual, popFitness, handle);
	gpuErrchk(cudaPeekAtLastError());
}

/// <summary>
/// Runs elite selection.
/// </summary>
/// <param name="populationFitness">The population fitness.</param>
/// <param name="orderedIndices">The ordered indices.</param>
void runEliteSelection(thrust::device_ptr<float> populationFitness, thrust::device_ptr<int> orderedIndices){

	sort_population(populationFitness, orderedIndices);
	gpuErrchk(cudaPeekAtLastError());

}

/// <summary>
/// Runs elite cellular selection.
/// </summary>
/// <param name="populationFitness">The population fitness.</param>
/// <param name="orderedIndices">The ordered indices.</param>
void runEliteCellularSelection(float populationFitness[], int selectedIndices[]){
#if (SELECTION == CELLULAR)
	cellularNeighborhood < POPULATION / POPULATIONS, POPULATION / POPULATIONS, COLS> << <POPULATIONS, POPULATION / POPULATIONS >> >(populationFitness, selectedIndices);
	gpuErrchk(cudaPeekAtLastError());
#endif
}

/// <summary>
/// Runs roulette selection.
/// </summary>
/// <param name="populationFitness">The population fitness.</param>
/// <param name="devicePtrPopulationFitness">A thrust device_ptr to population fitness.</param>
/// <param name="chosenIndices">The chosen indices.</param>
/// <param name="globalState">curandState.</param>
/// <param name="sumPopulationFitness">Sum of population(s) fitness.</param>
void runRouletteSelection(float * populationFitness, thrust::device_ptr<float> devicePtrPopulationFitness, int * chosenIndices, curandState* globalState, thrust::device_ptr<float> sumPopulationFitness){

	total_fitness(devicePtrPopulationFitness, sumPopulationFitness);

	rouletteSelection<POPULATION / POPULATIONS> << <(ELITE_MEMBERS + 511) / 512, 512 >> >(populationFitness, chosenIndices, sumPopulationFitness, globalState);
}

/// <summary>
/// Runs migration with the frequency defined by MIGRATION_FREQUENCY constant and moving ELEMENTS_TO_MIGRATE total migrations.
/// </summary>
/// <param name="currentGeneration">The current generation.</param>
/// <param name="migrationFrequency">The migration frequency.</param>
/// <param name="population">The population.</param>
/// <param name="chosenIndices">The chosen indices.</param>
void migrate(unsigned int currentGeneration, unsigned int migrationFrequency, network * population, int * chosenIndices){
	if (POPULATIONS > 1 && (currentGeneration%migrationFrequency) == 0){
		migrate<ELEMENTS_TO_MIGRATE / POPULATIONS, POPULATION / POPULATIONS> << <ELEMENTS_TO_MIGRATE / 2, 128 >> >(population, chosenIndices);
		gpuErrchk(cudaPeekAtLastError());
	}
}
__global__ void randomMigrationIndices(int migrationIndices[], curandState* globalState){
	const unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	if (index < POPULATION){
		migrationIndices[index] = (POPULATION / POPULATIONS) * generate(globalState);
	}
}


int main(void) {


	for (int e = 0; e < EXECUTIONS; e++){		// Runs EXECUTIONS independent runs

		float time;								// Holds time for current execution
		cudaEvent_t start, stop;				// CUDA Timers

		/*
		Timers creation
		*/
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventRecord(start));
		/*
		File for the output
		*/
		FILE *f;

		/*
		cuBLAS variables
		*/

		cudaError_t cudastat;
		cublasStatus_t stat;

		cublasHandle_t handle;
		stat = cublasCreate(&handle);

		/*
		Genetic algorithm arrays
		*/

		/*
		Population
		*/

		network *d_population;	// Network population

		HANDLE_ERROR(
			cudaMalloc(&d_population, POPULATION*sizeof(network)))
			;

		/*
		Fitness
		*/

		HANDLE_ERROR(
			cudaMemcpyToSymbol(LAMBDA_VALUES, &LAMBDA_HOST, sizeof(float)*POPULATIONS));		// Lambda values for each population

		char * d_goalLinks;					// Goal links
		HANDLE_ERROR(
			cudaMalloc(&d_goalLinks, TOTAL_LINKS*sizeof(char)));
		HANDLE_ERROR(
			cudaMemcpy(d_goalLinks, &GOAL_NETWORK_HOST.links, sizeof(char)*TOTAL_LINKS, cudaMemcpyHostToDevice));

		char * d_initNodes;					// Initial nodes
		HANDLE_ERROR(
			cudaMalloc(&d_initNodes, TOTAL_LINKS*sizeof(char)));
		HANDLE_ERROR(
			cudaMemcpy(d_initNodes, H_INIT_NODES, sizeof(char)*NODES, cudaMemcpyHostToDevice));

		char * d_goalNodes;					// Goal nodes
		HANDLE_ERROR(
			cudaMalloc(&d_goalNodes, NODES*sizeof(char)));
		HANDLE_ERROR(
			cudaMemcpy(d_goalNodes, &H_GOAL_NODES, sizeof(char)*NODES, cudaMemcpyHostToDevice));

		network h_bestIndividual;				// Best individual for each generation
		float h_bestFitness = 1.0;				// Fitness of the best individual

		float *d_populationFitness;				// Holds total population fitness for each generation

		HANDLE_ERROR(
			cudaMalloc(&d_populationFitness, POPULATION*sizeof(float)));
		thrust::device_ptr<float> device_ptr_fitness = thrust::device_pointer_cast(d_populationFitness);

		float* d_individualsLinkFitness;					// Link normalized fitness of each individual
		float* d_individualsNodeFitness;					// Node normalized fitness of each individual

		HANDLE_ERROR(
			cudaMalloc(&d_individualsLinkFitness, POPULATION*sizeof(float)));
		HANDLE_ERROR(
			cudaMalloc(&d_individualsNodeFitness, POPULATION*sizeof(float)));

		/*
		Vectors filled with ones used by cuBLAS matrix multiplication to sum all node/links distances.
		If we multiply a matrix by a ones vector, we will obtain the sum of all its rows.
		*/

		float * d_linksOnesVector, *d_nodesOnesVector;

		float* x_links = new float[TOTAL_LINKS];

		for (int i = 0; i < TOTAL_LINKS; i++)
		{
			x_links[i] = 1;
		}

		float* x_nodes = new float[NODES];

		for (int i = 0; i < NODES; i++)
		{
			x_nodes[i] = 1;
		}

		HANDLE_ERROR(
			cudaMalloc(&d_linksOnesVector, TOTAL_LINKS*sizeof(float)));
		HANDLE_ERROR(
			cudaMemcpy(d_linksOnesVector, x_links, TOTAL_LINKS*sizeof(float), cudaMemcpyHostToDevice));

		HANDLE_ERROR(
			cudaMalloc(&d_nodesOnesVector, NODES*sizeof(float)));
		HANDLE_ERROR(
			cudaMemcpy(d_nodesOnesVector, x_nodes, NODES*sizeof(float), cudaMemcpyHostToDevice));

		free(x_links);
		free(x_nodes);

		float *d_singleNodeDistances, *d_singleLinkDistances;	// Node and link distances PER NETWORK, that is, a[i] == 1 if i link/node value is different from the same node/link in goal network

		HANDLE_ERROR(
			cudaMalloc(&d_singleNodeDistances, POPULATION * NODES * sizeof(float)));
		HANDLE_ERROR(
			cudaMalloc(&d_singleLinkDistances, POPULATION * TOTAL_LINKS * sizeof(float)));

		int * d_chosenIndividuals;					// Indices of the chosen inviduals

		HANDLE_ERROR(
			cudaMalloc(&d_chosenIndividuals, ELITE_MEMBERS*sizeof(int)));
		/*************************************/
		/*************************************/
		/*************************************/
		int * d_migrationIndices;
		HANDLE_ERROR(
			cudaMalloc(&d_migrationIndices, ELITE_MEMBERS*sizeof(int)));
		/*************************************/
		/*************************************/

		/*************************************/
		thrust::device_ptr<int> dev_indices = thrust::device_pointer_cast(d_chosenIndividuals);

		float * d_sumFitness;					// Holds total fitness of each population
		HANDLE_ERROR(
			cudaMalloc(&d_sumFitness, POPULATIONS*sizeof(float)));

		thrust::device_ptr<float> dev_fitnessSum = thrust::device_pointer_cast(d_sumFitness);

		int *d_linkCrossPoints, *d_rulesCrossPoints;				// Arrays for random crossover indices
		HANDLE_ERROR(
			cudaMalloc(&d_linkCrossPoints, (ELITE_MEMBERS / 2)*sizeof(int)));
		HANDLE_ERROR(
			cudaMalloc(&d_rulesCrossPoints, (ELITE_MEMBERS / 2)*sizeof(int)));


		/*
		Misc
		*/

		curandState* devStates;					// cuRAND states

		HANDLE_ERROR(
			cudaMalloc(&devStates, POPULATION * 1024 * sizeof(curandState)));

		/** Streams **/
		cudaStream_t stream0, stream1, stream2, stream3, stream4;
		cudaStreamCreate(&stream0);
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		cudaStreamCreate(&stream3);
		cudaStreamCreate(&stream4);

		/** Calculate initial connectivity and save it in contant memory **/

		int h_initNullLinks = thrust::count(INIT_NETWORK_HOST.links, INIT_NETWORK_HOST.links + TOTAL_LINKS, 0);	// Null links in initial network

		const float h_initConnectivity = (TOTAL_LINKS - (float)h_initNullLinks) / (TOTAL_LINKS);				// Connectivity of initial network

		HANDLE_ERROR(
			cudaMemcpyToSymbol(INIT_CONNECTIVITY, &h_initConnectivity, sizeof(float)));							// Save it in constant memory

		/** Create a file to save algorithm's evolution **/

		char buf[0x100];
		_snprintf(buf, sizeof(buf), "P-Sexec%d%s_pob%dpops%d_MIGRs%d_gen%dfreq%d.csv", e, "CE", POPULATION, POPULATIONS, ELEMENTS_TO_MIGRATE / POPULATIONS, GENERATIONS, MIGRATION_FREQUENCY);

		f = fopen(buf, "w");
		if (f == NULL)
		{
			printf("Error opening file!\n");
			exit(1);
		}

		/** Print current configuration **/

		printParameters();

		/** Setup random number sequences **/

		setupCurandDiffSeed << < POPULATION * 2, 512 >> > (devStates);
		gpuErrchk(cudaPeekAtLastError());

		/** Generate initial population **/

		generateInitialPopulation << <(POPULATION + 1023) / 512, 512 >> >(d_population, devStates);
		gpuErrchk(cudaPeekAtLastError());

		/** Initial elementes indices: 0, 1, 2, 3... **/

		sequence << <(POPULATION + 1023) / 1024, 1024 >> >(d_chosenIndividuals);
		gpuErrchk(cudaPeekAtLastError())

			/** Random link crossover indeces **/

			generateLinkCrossPoints << < (ELITE_MEMBERS / 2 + 447) / 448, 448 >> >(d_linkCrossPoints, devStates);
		gpuErrchk(cudaPeekAtLastError());

		for (int i = 0; i < GENERATIONS && h_bestFitness != 0; i++){

			/** Evaluate population **/
			computeFitness(stream3, stream4, d_population, d_singleLinkDistances, d_linksOnesVector, d_individualsLinkFitness, handle, d_singleNodeDistances, d_nodesOnesVector, d_individualsNodeFitness, d_populationFitness, d_goalLinks, d_initNodes, d_goalNodes);

			/** Save best individual **/
			updateBestIndividual(d_population, d_populationFitness, i, f, &h_bestFitness, &h_bestIndividual, handle);
			gpuErrchk(cudaPeekAtLastError());

			/** Parent selection **/

			switch (SELECTION){
			case ELITE:
				runEliteSelection(device_ptr_fitness, dev_indices);
				break;
			case ROULETTE:
				runRouletteSelection(d_populationFitness, device_ptr_fitness, d_chosenIndividuals, devStates, dev_fitnessSum);
				break;
			case CELLULAR:
				runEliteCellularSelection(d_populationFitness, d_chosenIndividuals);
				break;
			}

			gpuErrchk(cudaPeekAtLastError());

			/** Migrate elements **/
			if (SELECTION == CELLULAR){
				randomMigrationIndices << <(POPULATION + 511) / 512, 512 >> > (d_migrationIndices, devStates);
				migrate(i, MIGRATION_FREQUENCY, d_population, d_migrationIndices);
			}
			else{
				migrate(i, MIGRATION_FREQUENCY, d_population, d_chosenIndividuals);
			}
			/** Show progress in console **/
			print_progress((float)i / (float)GENERATIONS, h_bestFitness);
			gpuErrchk(cudaPeekAtLastError());

			/** Crossover **/
			crossover<512, POPULATION / POPULATIONS, SELECTION> << <ELITE_MEMBERS / 2, (TOTAL_LINKS) / 2, 0, stream2 >> >(d_population, d_chosenIndividuals, d_linkCrossPoints);
			gpuErrchk(cudaPeekAtLastError());

			/** Mutation **/
			mutation << <(POPULATION + 32 * 5 + 1) / (32 * 5), 32 * 5, 0, stream0 >> >(d_population, devStates);
			gpuErrchk(cudaPeekAtLastError());

			/** Reset individual indices for next generation **/
			sequence << <(POPULATION + 1023) / 1024, 1024, 0, stream1 >> >(d_chosenIndividuals);
			gpuErrchk(cudaPeekAtLastError());

			/** Generate new crossover indices **/
			generateLinkCrossPoints << < (ELITE_MEMBERS / 2 + 447) / 448, 448, 0, stream2 >> >(d_linkCrossPoints, devStates);
			gpuErrchk(cudaPeekAtLastError());

			//	gpuErrchk( cudaDeviceSynchronize() );
		}

		/** Stop timers **/
		HANDLE_ERROR(cudaEventRecord(stop));
		HANDLE_ERROR(cudaEventSynchronize(stop));
		HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

		/** Write time and best individual **/
		fprintf(f, "\nTime to generate:  %3.1f ms \n", time);
		print_network_file(h_bestIndividual, f);

		/** Close file **/
		fclose(f);

		/** Free memory **/
		cudaFree(d_population);

		cudaFree(d_initNodes);
		cudaFree(d_goalLinks);
		cudaFree(d_goalNodes);

		cudaFree(d_populationFitness);

		cudaFree(d_individualsLinkFitness);
		cudaFree(d_individualsNodeFitness);

		cudaFree(d_linksOnesVector);
		cudaFree(d_nodesOnesVector);

		cudaFree(d_singleNodeDistances);
		cudaFree(d_singleLinkDistances);

		cudaFree(d_chosenIndividuals);
		cudaFree(d_migrationIndices);

		cudaFree(d_sumFitness);

		cudaFree(d_linkCrossPoints);
		cudaFree(d_rulesCrossPoints);

		cudaFree(devStates);

		/** Destroy streams **/

		cudaStreamDestroy(stream0);
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);
		cudaStreamDestroy(stream3);
		cudaStreamDestroy(stream4);

		cublasDestroy(handle);

	}

	cudaDeviceReset();
	return 0;
}
