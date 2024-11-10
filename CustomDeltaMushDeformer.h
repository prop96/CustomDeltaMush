#pragma once

#include <maya/MPxDeformerNode.h>
#include <maya/MDataBlock.h>
#include <maya/MItGeometry.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MVector.h>
#include <maya/MPointArray.h>
#include <vector>

class CustomDeltaMushDeformer : public MPxDeformerNode
{
public:
	CustomDeltaMushDeformer() = default;
	~CustomDeltaMushDeformer() = default;

	MStatus InitializeData(MObject& mesh);
	MStatus InitializeData(MObject& mesh, uint32_t smoothingIter, double smoothingAmount);
	void ApplyDeltaMush(const std::vector<MPoint>& skinned, std::vector<MPoint>& deformed, double envelope, double applyDeltaAmount) const;
	void SetSmoothingData(uint32_t iter, double amount);

	MStatus deform(MDataBlock& data, MItGeometry& iter, const MMatrix& l2w, unsigned int multiIdx) override;
	static void* creator();
	static MStatus initialize();

	static MTypeId id;
	inline static const MString nodeTypeName = "customDeltaMushDeformer";
	static MString pluginPath; ///< path where the loaded plugin exists

	// custom attributes
	static MObject rebind;
	static MObject smoothIterations;
	static MObject applyDelta;
	static MObject smoothAmount;

private:
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

	std::vector<PointData> m_pointData;
	bool m_isInitialized = false;
	SmoothingData m_smoothingData;

	void ComputeSmoothedPoints(const std::vector<MPoint>& src, std::vector<MPoint>& smoothed) const;
	void ComputeDelta(const std::vector<MPoint>& src, const std::vector<MPoint>& smoothed);
};