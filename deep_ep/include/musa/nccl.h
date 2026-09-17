#pragma once
#include<cuda_runtime.h>
#include<cuda_fp8.h>
#include<mccl.h>
#include<mccl_device.h>

/* MCCL compatibility */
// #define ncclResult_t mcclResult_t
#define ncclSuccess mcclSuccess
#define ncclUniqueId mcclUniqueId
#define ncclComm_t mcclComm_t
#define ncclCommProperties mcclCommProperties
#define ncclDevCommRequirements_t mcclDevCommRequirements_t
#define ncclDevComm_t mcclDevComm_t
#define ncclWindow_t mcclWindow_t

#define NCCL_COMM_PROPERTIES_INITIALIZER MCCL_COMM_PROPERTIES_INITIALIZER
#define NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER MCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
#define NCCL_GIN_TYPE_NONE MCCL_GIN_TYPE_NONE
#define NCCL_GIN_CONNECTION_RAIL MCCL_GIN_CONNECTION_RAIL
#define NCCL_GIN_CONNECTION_FULL MCCL_GIN_CONNECTION_FULL
#define NCCL_WIN_DEFAULT MCCL_WIN_DEFAULT
#define NCCL_WIN_STRICT_ORDERING MCCL_WIN_STRICT_ORDERING
#define NCCL_VERSION_CODE MCCL_VERSION_CODE
#define NCCL_VERSION MCCL_VERSION

#define ncclGetUniqueId mcclGetUniqueId
#define ncclCommInitRank mcclCommInitRank
#define ncclCommAbort mcclCommAbort
#define ncclGetVersion mcclGetVersion
#define ncclGetLastError mcclGetLastError
#define ncclCommQueryProperties mcclCommQueryProperties
#define ncclDevCommCreate mcclDevCommCreate
#define ncclDevCommDestroy mcclDevCommDestroy
#define ncclMemAlloc mcclMemAlloc
#define ncclMemFree mcclMemFree
#define ncclCommWindowRegister mcclCommWindowRegister
#define ncclCommWindowDeregister mcclCommWindowDeregister
#define ncclGetLsaDevicePointer mcclGetLsaDevicePointer

/* MCCL device compatibility */

#define ncclGin mcclGin
#define ncclTeam mcclTeam
#define ncclTeam_t mcclTeam_t
#define ncclTeamTagWorld mcclTeamTagWorld
#define ncclTeamTagLsa mcclTeamTagLsa
#define ncclTeamTagRail mcclTeamTagRail
#define ncclTeamWorld mcclTeamWorld
#define ncclTeamLsa mcclTeamLsa
#define ncclTeamRail mcclTeamRail
// #define ncclTeamRankIsMember mcclTeamRankIsMember
#define ncclCoopThread mcclCoopThread
#define ncclCoopWarp mcclCoopWarp
#define ncclGetLsaPointer mcclGetLsaPointer
#define ncclGin_VASignalAdd mcclGin_VASignalAdd
#define ncclGin_None mcclGin_None
#define ncclGin_SignalInc mcclGin_SignalInc
#define ncclGinSignal_t mcclGinSignal_t
#define ncclGinRequest_t mcclGinRequest_t
#define ncclGinResourceSharingMode mcclGinResourceSharingMode
#define NCCL_GIN_RESOURCE_SHARING_CTA MCCL_GIN_RESOURCE_SHARING_CTA
#define NCCL_GIN_RESOURCE_SHARING_GPU MCCL_GIN_RESOURCE_SHARING_GPU
// #define ncclGinOptFlagsDefault mcclGinOptFlagsDefault
// #define ncclGinOptFlagsAggregateRequests mcclGinOptFlagsAggregateRequests
#define ncclGin_SegmentDevice mcclGin_SegmentDevice
#define ncclGinGdakiGPUContext mcclGinGdakiGPUContext

namespace cuda = musa;

enum ncclGinOptFlags {
  ncclGinOptFlagsDefault = 0,
  ncclGinOptFlagsMaySkipCreditCheck = (1 << 0),
  ncclGinOptFlagsAggregateRequests = (1 << 1),
};