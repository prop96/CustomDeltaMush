#pragma once
#include <maya/MPointArray.h>
#include <maya/MFnMesh.h>

namespace DMUtil
{
	MStatus SmoothMesh(MObject& mesh, const MPointArray& original, MPointArray& smoothed);
	MStatus SmoothVertex(MPoint& smoothed, MItMeshVertex& itVertex, const MPointArray& points);
	MStatus SmoothVertex(MPoint& smoothed, const MPoint& original, const MIntArray& connected, const MPointArray& points);
	MStatus CreateDeltaMushMatrix(MMatrix& matrix, MItMeshVertex& itVertex, const MFnMesh& meshFn, const MPointArray& points);
}