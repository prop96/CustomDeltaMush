#include "DeltaMushUtil.h"
#include "MeshLaplacian.h"
#include <maya/MItMeshVertex.h>
#include <maya/MItMeshEdge.h>
#include <maya/MIntArray.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <cassert>
#include <set>

namespace DMUtil
{
	MStatus SmoothMesh(MObject& mesh, const MPointArray& original, MPointArray& smoothed, int smoothItr, double smoothAmount)
	{
		const unsigned int numVerts = original.length();
		CHECK_MSTATUS(smoothed.setLength(numVerts));

		// compute the smoothing matrix
		Eigen::SparseMatrix<double> B(numVerts, numVerts);
		bool isImplicit = false;
		MeshLaplacian::ComputeSmoothingMatrix(std::move(MItMeshEdge(mesh)), numVerts, smoothAmount, smoothItr, isImplicit, B);

		for (unsigned int idx = 0; idx < numVerts; idx++)
		{
			smoothed[idx] = MPoint(0, 0, 0, 0);

			for (unsigned int k = 0; k < numVerts; k++)
			{
				smoothed[idx] += B.coeff(k, idx) * original[k];
			}

			smoothed[idx].w = 1.0;
		}

		return MS::kSuccess;
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

	MStatus CreateDeltaMushMatrix(MObject& mesh, MMatrixArray& matrixArray)
	{
		MStatus returnStat;

		MFnMesh meshFn(mesh);

		// smooth 前の頂点を取得
		MPointArray originalPoints;
		meshFn.getPoints(originalPoints);

		// smooth 後の頂点を計算
		MPointArray smoothedPoints;
		SmoothMesh(mesh, originalPoints, smoothedPoints, 1, 1.0);

		// smooth 後の各頂点に対して、normal、tangent、bitangent を計算していく
		MItMeshVertex itVertex(mesh);
		matrixArray.setLength(0);
		for (itVertex.reset(); !itVertex.isDone(); itVertex.next())
		{
			// smooth 後の頂点位置
			int idx = itVertex.index();
			MPoint s = smoothedPoints[idx];

			// 隣接している頂点を取得
			MIntArray connectedVerts;
			itVertex.getConnectedVertices(connectedVerts);

			// smooth する前の法線を取得しておく
			MVector normalBefore;
			itVertex.getNormal(normalBefore);

			// 各 face に対して、新しい法線を計算し、それらの平均から法線を計算
			MVector n = MVector::zero;
			MIntArray connectedFaces;
			itVertex.getConnectedFaces(connectedFaces);
			for (const int& faceIdx : connectedFaces)
			{
				// face 内の頂点を取得
				MIntArray vertArray;
				meshFn.getPolygonVertices(faceIdx, vertArray);

				// この中から、隣接している2頂点のみ残して、ほかの要素を削除する
				for (int cvIdx = 0; cvIdx < vertArray.length(); )
				{
					bool found = false;
					for (const int& connectedVert : connectedVerts)
					{
						if (connectedVert == vertArray[cvIdx])
						{
							found = true;
							break;
						}
					}

					if (!found)
					{
						vertArray.remove(cvIdx);
					}
					else
					{
						++cvIdx;
					}
				}

				// 2要素だけ残っている想定
				assert(vertArray.length() == 2);

				// それらの外積から法線を計算
				MVector tmpNormal = ((smoothedPoints[vertArray[0]] - s) ^ (smoothedPoints[vertArray[1]] - s)).normal();

				// 法線の方向を計算
				MVector prevNormal;
				meshFn.getFaceVertexNormal(faceIdx, itVertex.index(), prevNormal);
				if (tmpNormal * prevNormal < 0.0f)
				{
					// NOTE: このやり方でうまくいかなかったら、MItMeshPolygon から Triangle を持ってくるか、
					// smooth 前の法線から掛け算の順序を推定する感じになる
					tmpNormal *= -1;
				}

				n += tmpNormal;
			}
			n /= connectedFaces.length();
			n.normalize();

			// tangent を計算する。方向は0番目の隣接頂点へ向く向きとする
			MVector t(smoothedPoints[connectedVerts[0]] - s);
			t = (t - (t * n) * n).normal();

			// binormal vector
			MVector b = (t ^ n).normal();

			// set value
			double tmp[4][4];
			{
				tmp[0][0] = t[0];  tmp[0][1] = t[1];  tmp[0][2] = t[2];  tmp[0][3] = t[3];
				tmp[1][0] = n[0];  tmp[1][1] = n[1];  tmp[1][2] = n[2];  tmp[1][3] = n[3];
				tmp[2][0] = b[0];  tmp[2][1] = b[1];  tmp[2][2] = b[2];  tmp[2][3] = b[3];
				tmp[3][0] = s[0];  tmp[3][1] = s[1];  tmp[3][2] = s[2];  tmp[3][3] = s[3];
			}
			matrixArray.append(tmp);
		}

		return returnStat;
	}

	MStatus CreateDeltaMushMatrix(MMatrix& matrix, MItMeshVertex& itVertex, const MFnMesh& meshFn, const MPointArray& points)
	{
		// FIXME: Normal, tangent, binormal は smmothed mesh に対して計算する必要がある

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