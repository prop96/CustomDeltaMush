#pragma once

#include <maya/MPxDeformerNode.h>
#include <maya/MDataBlock.h>
#include <maya/MItGeometry.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <vector>

struct point_data
{
	MIntArray neighbours;
	MVectorArray delta;
	int size;
	double deltaLen;
};

class CustomDeltaMushDeformer : public MPxDeformerNode
{
public:
	CustomDeltaMushDeformer();

	MStatus compute(const MPlug& plug, MDataBlock& data) override;
	MStatus deform(MDataBlock& data, MItGeometry& iter, const MMatrix& l2w, unsigned int multiIdx) override;
	static void* creator();
	static MStatus initialize();

public:
	static MObject deltaMushMatrix;

	static MTypeId id;

	inline static const MString nodeTypeName = "customDeltaMushDeformer";

public:
	static MObject rebind;
	static MObject smoothIterations;
	static MObject applyDelta;
	static MObject smoothAmount;
	static MObject globalScale;

private:
	MPointArray targetPos;
	std::vector<point_data> dataPoints;
	bool initialized;

	MStatus initData(MObject& mesh, int iters);
	void averageRelax(const MPointArray& source, MPointArray& target, int smoothIter, double smoothAmount);
	void computeDelta(MPointArray& source, MPointArray& target);
	void rebindData(MObject& mesh, int iter, double amount);
};