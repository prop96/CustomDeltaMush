#include "CustomDeltaMushDeformer.h"
#include "DeltaMushUtil.h"
#include <maya/MFnUnitAttribute.h>
#include <maya/MDistance.h>
#include <maya/MPoint.h>
#include <maya/MItMeshVertex.h>
#include <maya/MPointArray.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMatrixArrayData.h>
#include <maya/MFnMatrixAttribute.h>
#include <cassert>

/*
* select object and type
*   deformer -type SwirlDeformer
*/

// instancing the static fields
MTypeId CustomDeltaMushDeformer::id(0x80095);
MObject CustomDeltaMushDeformer::deltaMushMatrix;

MStatus CustomDeltaMushDeformer::compute(const MPlug& plug, MDataBlock& data)
{
	MString info = plug.info();
	cout << info << endl;

	return MPxDeformerNode::compute(plug, data);
}

MStatus CustomDeltaMushDeformer::deform(MDataBlock& data, MItGeometry& iter, const MMatrix& localToWorld, unsigned int multiIdx)
{
	MStatus returnStat;

	// get the attribute instance of input[0].inputGeometry
	MObject skinnedMesh = data.inputArrayValue(input).inputValue().child(inputGeom).data();

	// get all the positions of the vertices
	MFnMesh meshFn(skinnedMesh);
	MPointArray skinnedPoints;
	meshFn.getPoints(skinnedPoints);

	// get the deltaMushMatrix plug
	MArrayDataHandle dmArrayHandle = data.inputArrayValue(deltaMushMatrix, &returnStat);
	CHECK_MSTATUS(returnStat);
	int num = dmArrayHandle.elementCount();
	cout << num << endl;

	MPointArray deformed;
	deformed.setLength(num);

	MItMeshVertex itVertex(skinnedMesh);
	for (itVertex.reset(); !itVertex.isDone(); itVertex.next())
	{
		CHECK_MSTATUS(dmArrayHandle.jumpToElement(itVertex.index()));

		MMatrix inv = dmArrayHandle.inputValue(&returnStat).asMatrix().inverse();
		CHECK_MSTATUS(returnStat);

		MMatrix mat;
		CHECK_MSTATUS(DMUtil::CreateDeltaMushMatrix(mat, itVertex, meshFn, skinnedPoints));

		MPoint pos = itVertex.position();
		pos = pos * inv * mat;
		deformed[itVertex.index()] = pos;
	}

	for (iter.reset(); !iter.isDone(); iter.next())
	{
		iter.setPosition(deformed[iter.index()]);
	}

	return returnStat;
}

void* CustomDeltaMushDeformer::creator()
{
	return new CustomDeltaMushDeformer();
}

MStatus CustomDeltaMushDeformer::initialize()
{
	MStatus returnStat;

	//MFnTypedAttribute tAttr;
	//deltaMushMatrix = tAttr.create("deltaMushMatrix", "dmMat", MFnData::kMatrixArray, MObject::kNullObj, &returnStat);

	MFnMatrixAttribute mAttr;
	deltaMushMatrix = mAttr.create("deltaMushMatrix", "dmMat", MFnMatrixAttribute::kDouble, &returnStat);
	CHECK_MSTATUS(returnStat);
	CHECK_MSTATUS(mAttr.setArray(true));

	CHECK_MSTATUS(addAttribute(deltaMushMatrix));

	CHECK_MSTATUS(attributeAffects(deltaMushMatrix, outputGeom));

	return returnStat;
}
