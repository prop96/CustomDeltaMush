#pragma once

#include <maya/MPxDeformerNode.h>
#include <maya/MDataBlock.h>
#include <maya/MItGeometry.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>

class CustomDeltaMushDeformer : public MPxDeformerNode
{
public:
	MStatus compute(const MPlug& plug, MDataBlock& data) override;
	MStatus deform(MDataBlock& data, MItGeometry& iter, const MMatrix& l2w, unsigned int multiIdx) override;
	static void* creator();
	static MStatus initialize();

public:
	static MObject deltaMushMatrix;

	static MTypeId id;

	inline static const MString nodeTypeName = "customDeltaMushDeformer";
};