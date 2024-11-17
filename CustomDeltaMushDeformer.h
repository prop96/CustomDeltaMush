#pragma once

#include "DeltaMushBindMeshData.h"
#include <maya/MPxDeformerNode.h>
#include <vector>

class CustomDeltaMushDeformer : public MPxDeformerNode
{
public:
	CustomDeltaMushDeformer() = default;
	~CustomDeltaMushDeformer() = default;

	MStatus deform(MDataBlock& data, MItGeometry& iter, const MMatrix& l2w, unsigned int multiIdx) override;
	static void* creator();
	static MStatus initialize();

	// custom attributes
	static MObject rebind;
	static MObject smoothIterations;
	static MObject applyDelta;
	static MObject smoothAmount;

	// static fields
	static MTypeId id;
	inline static const MString nodeTypeName = "customDeltaMushDeformer";
	static MString pluginPath; ///< path where the loaded plugin exists

private:
	DeltaMushBindMeshData m_bindMeshData;

	void ApplyDeltaMush(const std::vector<MPoint>& skinned, std::vector<MPoint>& deformed, double envelope, double applyDeltaAmount) const;
};