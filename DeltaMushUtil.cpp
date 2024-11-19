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

	void ComputeSmoothedPoints(
		const std::vector<MPoint>& src,
		std::vector<MPoint>& smoothed,
		const SmoothingData& smoothingData,
		const std::vector<std::vector<int32_t>>& neighbourIndicesAll)
	{
#if 0
		// verify Laplacian Smoothing
		CHECK_MSTATUS(DMUtil::SmoothMesh(mesh, source, target, smoothItr, smoothAmount));
		return;
#endif

		const uint32_t numVerts = src.size();
		std::vector<MPoint> srcCopy(numVerts);

		smoothed.resize(numVerts);
		std::copy(src.begin(), src.end(), smoothed.begin());

		for (uint32_t itr = 0; itr < smoothingData.Iter; itr++)
		{
			srcCopy.swap(smoothed);

			for (uint32_t vertIdx = 0; vertIdx < numVerts; vertIdx++)
			{
				const std::vector<int32_t>& neighbourVertexIndices = neighbourIndicesAll[vertIdx];

				// 隣接頂点の平均としてスムージング
				MVector smoothedPos = MVector::zero;
				for (const int neighbourIdx : neighbourVertexIndices)
				{
					smoothedPos += srcCopy[neighbourIdx];
				}
				smoothedPos *= 1.0 / double(neighbourVertexIndices.size());

				smoothed[vertIdx] = srcCopy[vertIdx] + (smoothedPos - srcCopy[vertIdx]) * smoothingData.Amount;
			}
		}
	}

	MMatrix ComputeTangentMatrix(const MPoint& pos, const MPoint& posNeighbor0, const MPoint& posNeighbor1)
	{
		// 注目している頂点と隣接頂点の作る三角形ポリゴンを考えて、tangent matrix を作る
		MVector v0 = posNeighbor0 - pos;
		MVector v1 = posNeighbor1 - pos;

		v0.normalize();
		v1.normalize();

		// tangent, normal, binormal
		MVector t = v0;
		MVector n = t ^ v1;
		MVector b = n ^ t;

		MMatrix mat = MMatrix();
		{
			mat[0][0] = t.x;
			mat[0][1] = t.y;
			mat[0][2] = t.z;
			mat[0][3] = 0;
			mat[1][0] = b.x;
			mat[1][1] = b.y;
			mat[1][2] = b.z;
			mat[1][3] = 0;
			mat[2][0] = n.x;
			mat[2][1] = n.y;
			mat[2][2] = n.z;
			mat[2][3] = 0;
			mat[3][0] = 0;
			mat[3][1] = 0;
			mat[3][2] = 0;
			mat[3][3] = 1;
		}

		return mat;
	}

	void MPointArrayToVector(std::vector<MPoint>& vec, const MPointArray& ptArr)
	{
		const uint32_t num = ptArr.length();
		vec.resize(num);
		for (uint32_t idx = 0; idx < num; idx++)
		{
			vec[idx] = ptArr[idx];
		}
	}

	void VectorToMPointArray(MPointArray& ptArr, const std::vector<MPoint>& vec)
	{
		const uint32_t num = vec.size();
		ptArr.setLength(num);
		for (uint32_t idx = 0; idx < num; idx++)
		{
			ptArr[idx] = vec[idx];
		}
	}
}