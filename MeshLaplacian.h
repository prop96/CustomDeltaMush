#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <maya/MItMeshEdge.h>

class MeshLaplacian
{
public:
	static Eigen::SparseMatrix<double> TestMatrix(int);

	static void DiagonalizeGenSparseMatrix(const Eigen::SparseMatrix<double>& Mat, const std::string filepath);

	static void GetDiagonalizationResult(
		Eigen::VectorXcd& eigVals,
		Eigen::MatrixXcd& eigVecs,
		const std::vector<unsigned int>& indices,
		const int numVertices,
		const std::string filepath);

	/// <summary>
	/// Compute Normalized Laplacian
	/// </summary>
	/// <param name="indices"></param>
	/// <param name="numVertices"></param>
	/// <returns></returns>
	static void ComputeLaplacian(
		MItMeshEdge& itEdge,
		const int numVertices,
		Eigen::SparseMatrix<double>& laplacian);

	static void ComputeSmoothingMatrix(
		const std::vector<unsigned int>& indices,
		const int numVertices,
		const std::string filepath,
		double lambda,
		int p);

	static void ComputeSmoothingMatrix(
		MItMeshEdge& itEdge,
		const int numVertices,
		double lambda,
		int p,
		bool isImplicit,
		Eigen::SparseMatrix<double>& B);
};