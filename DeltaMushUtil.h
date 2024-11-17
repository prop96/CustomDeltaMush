#pragma once
#include <maya/MPointArray.h>
#include <maya/MFnMesh.h>
#include <maya/MMatrix.h>
#include <vector>


// data used to compute DeltaMush for each point
struct PointData
{
	std::vector<int32_t> NeighbourIndices; ///< indices of neighbouring points
	double DeltaLength = 0.0;              ///< distance b/w original pos and smoothed pos
	std::vector<MVector> Delta;            ///< delta computed in tangent space of triangles with neighbouring points
};

// parameters for laplacian smoothing
struct SmoothingData {
	uint32_t Iter = 0;   ///< smooth iterations
	double Amount = 1.0; ///< smoothing amount [0.0 ~ 1.0]
};

namespace DMUtil
{
	MStatus SmoothMesh(MObject& mesh, const MPointArray& original, MPointArray& smoothed, int smoothItr, double smoothAmount);

	void ComputeSmoothedPoints(const std::vector<MPoint>& src, std::vector<MPoint>& smoothed, const SmoothingData& smoothingData, const std::vector<PointData>& pointDataArray);

	MMatrix ComputeTangentMatrix(const MPoint& pos, const MPoint& posNeighbor0, const MPoint& posNeighbor1);

	void MPointArrayToVector(std::vector<MPoint>& vec, const MPointArray& ptArr);

	void VectorToMPointArray(MPointArray& ptArr, const std::vector<MPoint>& vec);
}