#pragma once

#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MPxNode.h>
#include <maya/MDataBlock.h>


class PrecomputeCustomSkinningNode : public MPxNode
{
public:
	virtual MStatus compute(const MPlug& plug, MDataBlock& data);
	static void* creator();
	static MStatus initialize();

public:
	// attributes
	static MObject originalGeometry;
	static MObject deltaMushMatrix;

	static MTypeId id;

	inline static const MString nodeTypeName = "precomputeCustomSkinning";
};