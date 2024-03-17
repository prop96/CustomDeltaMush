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

		// smooth �O�̒��_���擾
		MPointArray originalPoints;
		meshFn.getPoints(originalPoints);

		// smooth ��̒��_���v�Z
		MPointArray smoothedPoints;
		SmoothMesh(mesh, originalPoints, smoothedPoints, 1, 1.0);

		// smooth ��̊e���_�ɑ΂��āAnormal�Atangent�Abitangent ���v�Z���Ă���
		MItMeshVertex itVertex(mesh);
		matrixArray.setLength(0);
		for (itVertex.reset(); !itVertex.isDone(); itVertex.next())
		{
			// smooth ��̒��_�ʒu
			int idx = itVertex.index();
			MPoint s = smoothedPoints[idx];

			// �אڂ��Ă��钸�_���擾
			MIntArray connectedVerts;
			itVertex.getConnectedVertices(connectedVerts);

			// smooth ����O�̖@�����擾���Ă���
			MVector normalBefore;
			itVertex.getNormal(normalBefore);

			// �e face �ɑ΂��āA�V�����@�����v�Z���A�����̕��ς���@�����v�Z
			MVector n = MVector::zero;
			MIntArray connectedFaces;
			itVertex.getConnectedFaces(connectedFaces);
			for (const int& faceIdx : connectedFaces)
			{
				// face ���̒��_���擾
				MIntArray vertArray;
				meshFn.getPolygonVertices(faceIdx, vertArray);

				// ���̒�����A�אڂ��Ă���2���_�̂ݎc���āA�ق��̗v�f���폜����
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

				// 2�v�f�����c���Ă���z��
				assert(vertArray.length() == 2);

				// �����̊O�ς���@�����v�Z
				MVector tmpNormal = ((smoothedPoints[vertArray[0]] - s) ^ (smoothedPoints[vertArray[1]] - s)).normal();

				// �@���̕������v�Z
				MVector prevNormal;
				meshFn.getFaceVertexNormal(faceIdx, itVertex.index(), prevNormal);
				if (tmpNormal * prevNormal < 0.0f)
				{
					// NOTE: ���̂����ł��܂������Ȃ�������AMItMeshPolygon ���� Triangle �������Ă��邩�A
					// smooth �O�̖@������|���Z�̏����𐄒肷�銴���ɂȂ�
					tmpNormal *= -1;
				}

				n += tmpNormal;
			}
			n /= connectedFaces.length();
			n.normalize();

			// tangent ���v�Z����B������0�Ԗڂ̗אڒ��_�֌��������Ƃ���
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
		// FIXME: Normal, tangent, binormal �� smmothed mesh �ɑ΂��Čv�Z����K�v������

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