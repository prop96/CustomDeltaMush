#include "MeshLaplacian.h"
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>

#include <iomanip>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <iostream>
#include <set>
#include <array>

typedef Eigen::Triplet<double> Trp;
const double err = 1e-12;


Eigen::SparseMatrix<double> MeshLaplacian::TestMatrix(int seed = -1)
{
    std::vector<Trp> tripletVec;

    if (seed == -1)
    {
        // case with complex eigen values and vectors
        tripletVec.push_back(Trp(0, 3, -1));
        tripletVec.push_back(Trp(3, 0, 1));
        tripletVec.push_back(Trp(1, 1, 1));
        tripletVec.push_back(Trp(2, 2, -1));
    }
    else
    {
        tripletVec.push_back(Trp(0, 3, 1));
        tripletVec.push_back(Trp(1, 2, 2));
        tripletVec.push_back(Trp(2, 1, 3));
        tripletVec.push_back(Trp(3, 0, 2));
    }

    Eigen::SparseMatrix<double> mat(4, 4);
    mat.setFromTriplets(tripletVec.begin(), tripletVec.end());
    return mat;
}

void MeshLaplacian::DiagonalizeGenSparseMatrix(const Eigen::SparseMatrix<double>& Mat, const std::string filepath)
{
    int matSize = Mat.rows();
    if (matSize != Mat.cols())
    {
        std::cerr << "Error: only regular matrix can be diagonalized." << std::endl;
        return;
    }

    // decompose the process into two stages: compute larger half eigen values, and then smaller half
    int numEigs = matSize / 2;
    Spectra::SparseGenMatProd<double> op(Mat);

    // First, compute the larger half eigen values
    Spectra::GenEigsSolver<Spectra::SparseGenMatProd<double>> solverLargerHalf(op, numEigs + matSize % 2, matSize);
    solverLargerHalf.init();
    solverLargerHalf.compute(Spectra::SortRule::LargestReal);
    if (solverLargerHalf.info() != Spectra::CompInfo::Successful)
    {
        std::cerr << "Error: failed to compute larger half eigen values" << std::endl;
        return;
    }

    // Next, compute the smaller half eigen values
    Spectra::GenEigsSolver<Spectra::SparseGenMatProd<double>> solverSmallerHalf(op, numEigs, matSize);
    solverSmallerHalf.init();
    solverSmallerHalf.compute(Spectra::SortRule::SmallestReal);
    if (solverSmallerHalf.info() != Spectra::CompInfo::Successful)
    {
        std::cerr << "Error: failed to compute smaller half eigen values" << std::endl;
        return;
    }

    // Save the results in a csv file
    std::ofstream file;
    file.open(filepath, std::ios::out);
    std::cout << std::fixed;

    // write eigen values
    Eigen::VectorXcd eigVals(matSize);
    eigVals << solverLargerHalf.eigenvalues(), solverSmallerHalf.eigenvalues();
    for (size_t i = 0; i < matSize; i++)
    {
        file << std::setprecision(12) << eigVals[i];
        if (i == matSize - 1)
        {
            file << std::endl;
            break;
        }
        file << " ";
    }

    // write eigen vectors
    Eigen::MatrixXcd eigVecs(matSize, matSize);
    eigVecs << solverLargerHalf.eigenvectors(), solverSmallerHalf.eigenvectors();
    for (size_t i = 0; i < matSize; i++)
    {
        for (size_t j = 0; j < matSize; j++)
        {
            file << std::setprecision(12) << eigVecs.coeff(i, j);
            if (j == matSize - 1)
            {
                file << std::endl;
                break;
            }
            file << " ";
        }
    }
    file.close();

    // check
    //std::cout << "original mat:" << std::endl << Mat << std::endl;

    //std::cout << "eigvals: " << std::endl << eigVals << std::endl;
    //std::cout << "eigVecs: " << std::endl << eigVecs << std::endl;

    //Eigen::MatrixXcd inv = eigVecs.inverse();
    //Eigen::MatrixXcd original = eigVecs * (eigVals.asDiagonal()) * inv;
    //std::cout << "original:" << std::endl << original << std::endl;
}

void MeshLaplacian::GetDiagonalizationResult(
    Eigen::VectorXcd& eigVals,
    Eigen::MatrixXcd& eigVecs,
    const std::vector<unsigned int>& indices,
    const int numVertices,
    const std::string filepath)
{
    // if still not computed eigen vals and vectors, save it as the csv file
    if (!std::filesystem::exists(filepath))
    {
        Eigen::SparseMatrix<double> laplacian;
        //ComputeLaplacian(indices, numVertices, laplacian);
        DiagonalizeGenSparseMatrix(laplacian, filepath);
    }

    // read eigen values and vectors from csv file
    std::ifstream file;
    file.open(filepath, std::ios::in);
    std::string buf;

    // get eigen values from the first line
    if (std::getline(file, buf))
    {
        std::istringstream ss(buf);
        Eigen::dcomplex c;
        int idx = 0;
        while (ss >> c)
        {
            eigVals[idx++] = c;
        }
    }

    // get eigen vectors
    int row = 0;
    while (std::getline(file, buf))
    {
        std::istringstream ss(buf);
        Eigen::dcomplex c;
        int col = 0;
        while (ss >> c)
        {
            eigVecs(row, col++) = c;
        }
        row++;
    }
}

void MeshLaplacian::ComputeLaplacian(MItMeshEdge& itEdge, const int numVertices, Eigen::SparseMatrix<double>& laplacian)
{
    // generate the normalized Laplacian Matrix
    unsigned int matSize = numVertices;
    std::vector<Trp> tripletVec;
    Eigen::VectorXd degrees = Eigen::VectorXd::Zero(matSize);
    std::set<std::pair<int, int>> countedEdges;
    for (itEdge.reset(); !itEdge.isDone(); itEdge.next())
    {
        // get the vertices in the edge
        int idx0 = itEdge.index(0), idx1 = itEdge.index(1);

        auto edge = std::pair<int, int>(idx0, idx1);
        if (countedEdges.find(edge) == countedEdges.end())// ‚±‚Ì if •¶‚Í•s—v
        {
            degrees[idx0] += 1;
            degrees[idx1] += 1;

            tripletVec.push_back(Trp(idx0, idx1, 1));
            tripletVec.push_back(Trp(idx1, idx0, 1));

            countedEdges.emplace(std::move(edge));
        }
        else
        {
            std::cout << "‚±‚±‚É‚Í—ˆ‚È‚¢‚æ‚Ë‚¥" << std::endl;
        }
    }

    // Adjacency Matrix
    Eigen::SparseMatrix<double> A(matSize, matSize);
    A.setFromTriplets(tripletVec.begin(), tripletVec.end());

    //std::cout << "A: " << A << std::endl;

    // Inverse Degree Matrix
    Eigen::SparseMatrix<double> Dinv = Eigen::SparseMatrix<double>(degrees.cwiseInverse().asDiagonal());

    //std::cout << "Dinv: " << Dinv << std::endl;

    // Identity Matrix
    Eigen::SparseMatrix<double> Identity(matSize, matSize);
    Identity.setIdentity();

    // Normalized Laplacian
    laplacian = Identity - A * Dinv;

    //std::cout << "Laplacian: " << laplacian << std::endl;
}

void MeshLaplacian::ComputeSmoothingMatrix(
    const std::vector<unsigned int>& indices,
    const int numVertices,
    const std::string filepath,
    double lambda,
    int p)
{
    Eigen::VectorXcd eigVals(numVertices);
    Eigen::MatrixXcd eigVecs(numVertices, numVertices);

    GetDiagonalizationResult(eigVals, eigVecs, indices, numVertices, filepath);

    // compute B = (I + lambda*L)^{-p} = V * (I + lambda*D)^{-p} * V^{-1}  where L = V * D * V^{-1}
    Eigen::VectorXd diag(numVertices);
    for (size_t i = 0; i < numVertices; i++)
    {
        // assume all eigen values are real
        diag[i] = eigVals[i].real();//std::pow(1 + lambda * eigVals[i].real(), -p);
    }

    Eigen::MatrixXcd B = eigVecs /** diag.asDiagonal()*/ * eigVecs.inverse();

    //std::cout << "B:" << std::endl << B << std::endl;
    for (size_t i = 0; i < B.rows(); i++)
    {
        for (size_t j = 0; j < B.cols(); j++)
        {
            if (std::abs(B.coeff(i, j).imag()) > err)
            {
                std::cout << B.coeff(i, j) << std::endl;
            }
        }
    }
}

void MeshLaplacian::ComputeSmoothingMatrix(
    MItMeshEdge& itEdge,
    const int numVertices,
    double lambda,
    int p,
    bool isImplicit,
    Eigen::SparseMatrix<double>& B)
{
    Eigen::SparseMatrix<double> laplacian;
    ComputeLaplacian(itEdge, numVertices, laplacian);

    Eigen::SparseMatrix<double> Identity(numVertices, numVertices);
    Identity.setIdentity();

    Eigen::SparseMatrix<double> seed = Identity;
    if (isImplicit)
    {
        seed += lambda * laplacian;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> inverseSolver;
        inverseSolver.compute(seed);
        if (inverseSolver.info() != Eigen::Success)
        {
            return;
        }
        seed = inverseSolver.solve(Identity);
        if (inverseSolver.info() != Eigen::Success)
        {
            return;
        }
    }
    else
    {
        seed -= lambda * laplacian;
    }

    B = Identity;
    for (int i = 0; p!= 0 ; i++)
    {
        if (i > 0)
        {
            seed = seed * seed;
        }

        if (p & (1<<i))
        {
            B = B * seed;
            p &= ~(1 << i);
        }
    }
}
