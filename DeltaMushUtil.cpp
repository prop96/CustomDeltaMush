#include "DeltaMushUtil.h"
#include <maya/MItMeshVertex.h>
#include <maya/MIntArray.h>
#include <maya/MMatrix.h>
#include <cassert>

namespace DMUtil
{
	MStatus SmoothMesh(MObject& mesh, const MPointArray& original, MPointArray& smoothed)
	{
		MStatus returnStat;

		const unsigned int numVerts = original.length();
		returnStat = smoothed.setLength(numVerts);

		MItMeshVertex itVertex(mesh);
		for (itVertex.reset(); !itVertex.isDone(); itVertex.next())
		{
			MPoint pos(0, 0, 0);
			SmoothVertex(pos, itVertex, original);
			smoothed[itVertex.index()] = pos;
		}

		return returnStat;
	}

	MStatus SmoothVertex(MPoint& smoothed, MItMeshVertex& itVertex, const MPointArray& points)
	{
		MIntArray connected;
		itVertex.getConnectedVertices(connected);

		return SmoothVertex(smoothed, points[itVertex.index()], connected, points);
	}

	MStatus SmoothVertex(MPoint& smoothed, const MPoint& original, const MIntArray& connected, const MPointArray& points)
	{
		smoothed = MPoint(0, 0, 0, 1);

		const unsigned int numConnected = connected.length();
		if (numConnected == 0)
		{
			smoothed = original;
			return MS::kSuccess;
		}

		for (const int vidx : connected)
		{
			smoothed += points[vidx];
		}
		smoothed = smoothed / numConnected;
		return MS::kSuccess;
	}

	MStatus CreateDeltaMushMatrix(MMatrix& matrix, MItMeshVertex& itVertex, const MFnMesh& meshFn, const MPointArray& points)
	{
		// FIXME: Normal, tangent, binormal ‚Í smmothed mesh ‚É‘Î‚µ‚ÄŒvŽZ‚·‚é•K—v‚ª‚ ‚é

		// normal vector
		MVector n;
		itVertex.getNormal(n);
		n.normalize();

		// tangent vector
		MIntArray connected;
		itVertex.getConnectedVertices(connected);
		MPoint neighbor;
		meshFn.getPoint(connected[0], neighbor);
		MVector t(neighbor - itVertex.position());
		t = (t - (t * n) * n).normal();

		// binormal vector
		MVector b = t ^ n;

		// smoothed position
		MPoint s;
		DMUtil::SmoothVertex(s, itVertex, points);

		// set value
		double tmp[4][4];
		{
			tmp[0][0] = t[0];  tmp[0][1] = t[1];  tmp[0][2] = t[2];  tmp[0][3] = t[3];
			tmp[1][0] = n[0];  tmp[1][1] = n[1];  tmp[1][2] = n[2];  tmp[1][3] = n[3];
			tmp[2][0] = b[0];  tmp[2][1] = b[1];  tmp[2][2] = b[2];  tmp[2][3] = b[3];
			tmp[3][0] = s[0];  tmp[3][1] = s[1];  tmp[3][2] = s[2];  tmp[3][3] = s[3];
		}
		matrix = tmp;

		return MS::kSuccess;
	}
}