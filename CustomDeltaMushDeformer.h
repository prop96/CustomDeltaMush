#pragma once

#include "DeltaMushBindMeshData.h"
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

	void ApplyDeltaMush(const std::vector<MPoint>& skinned, std::vector<MPoint>& deformed, double envelope, double applyDeltaAmount) const;

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
	DeltaMushBindMeshData m_bindMeshData;
};