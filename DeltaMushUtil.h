#pragma once
#include <maya/MPointArray.h>
#include <maya/MFnMesh.h>
#include <maya/MMatrix.h>
#include <vector>

// parameters for laplacian smoothing
struct SmoothingData {
	uint32_t Iter = 0;   ///< smooth iterations
	double Amount = 1.0; ///< smoothing amount [0.0 ~ 1.0]
};

namespace DMUtil
{
	MStatus SmoothMesh(MObject& mesh, const MPointArray& original, MPointArray& smoothed, int smoothItr, double smoothAmount);

	void ComputeSmoothedPoints(
		const std::vector<MPoint>& src,
		std::vector<MPoint>& smoothed,
		const SmoothingData& smoothingData,
		const std::vector<uint32_t>& startIndices,
		const std::vector<int32_t>& neighbourIndices);

	MMatrix ComputeTangentMatrix(const MPoint& pos, const MPoint& posNeighbor0, const MPoint& posNeighbor1);

	void MPointArrayToVector(std::vector<MPoint>& vec, const MPointArray& ptArr);

	void VectorToMPointArray(MPointArray& ptArr, const std::vector<MPoint>& vec);
}