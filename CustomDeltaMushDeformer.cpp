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
#include <maya/MFnNumericAttribute.h>
#include <maya/MGlobal.h>
#include <cassert>

/*
* select object and type
*   deformer -type SwirlDeformer
*/

#define SMALL (float)1e-6

// instancing the static fields
MTypeId CustomDeltaMushDeformer::id(0x80095);
MObject CustomDeltaMushDeformer::deltaMushMatrix;

MObject CustomDeltaMushDeformer::rebind;
MObject CustomDeltaMushDeformer::referenceMesh;
MObject CustomDeltaMushDeformer::iterations;
MObject CustomDeltaMushDeformer::applyDelta;
MObject CustomDeltaMushDeformer::amount;
MObject CustomDeltaMushDeformer::globalScale;


CustomDeltaMushDeformer::CustomDeltaMushDeformer()
	: initialized(false)
{
	targetPos.setLength(0);
}

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

	MMatrixArray mats;
	DMUtil::CreateDeltaMushMatrix(skinnedMesh, mats);

	MItMeshVertex itVertex(skinnedMesh);
	for (itVertex.reset(); !itVertex.isDone(); itVertex.next())
	{
		CHECK_MSTATUS(dmArrayHandle.jumpToElement(itVertex.index()));

		MMatrix inv = dmArrayHandle.inputValue(&returnStat).asMatrix().inverse();
		CHECK_MSTATUS(returnStat);

		MMatrix mat = mats[itVertex.index()];
		//CHECK_MSTATUS(DMUtil::CreateDeltaMushMatrix(mat, itVertex, meshFn, skinnedPoints));

		MPoint pos = itVertex.position();
		pos = pos * inv * mat;
		deformed[itVertex.index()] = pos;
	}

	for (iter.reset(); !iter.isDone(); iter.next())
	{
		//iter.setPosition(deformed[iter.index()]);
	}


	{
		MPlug refMeshPlug(thisMObject(), referenceMesh);
		if (!refMeshPlug.isConnected())
		{
			std::cout << "ref mesh not connected" << std::endl;
			//return MS::kNotImplemented;
		}

		// getting needed data
		MObject referenceMeshV = skinnedMesh;//data.inputValue(referenceMesh).asMesh();
		double envelopeV = data.inputValue(envelope).asFloat();
		int iterationsV = data.inputValue(iterations).asInt();
		double applyDeltaV = data.inputValue(applyDelta).asDouble();
		double amountV = data.inputValue(amount).asDouble();
		bool rebindV = data.inputValue(rebind).asBool();
		double globalScaleV = data.inputValue(globalScale).asDouble();

		// extracting the points
		MPointArray pos;
		iter.allPositions(pos, MSpace::kWorld);

		if (!initialized || rebindV)
		{
			// binding the mesh
			rebindData(referenceMeshV, iterationsV, amountV);
			initialized = true;
		}

		if (envelopeV < SMALL)
		{
			return MS::kSuccess;
		}

		int size = iter.exactCount();
		int i, n;
		MVector delta, v1, v2, cross;

		double weight;
		MMatrix mat;
		MPointArray final;

		// here we perform the smooth
		averageRelax(pos, targetPos, iterationsV, amountV);
		if (iterationsV == 0)
		{
			return MS::kSuccess;
		}
		else
		{
			final.copy(targetPos);
		}

		// loopint the vertices, we know neet to re-apply the delta
		for (i = 0; i < size; i++)
		{
			// zeroing out the vector
			delta = MVector::zero;
			if (applyDeltaV >= SMALL)
			{
				// looping the neighbours
				for (n = 0; n < dataPoints[i].size-1; n++)
				{
					// extracting the next two neighbours and compute the vector
					v1 = targetPos[dataPoints[i].neighbours[n]] - targetPos[i];
					v2 = targetPos[dataPoints[i].neighbours[n + 1]] - targetPos[i];

					// normalizing
					v2.normalize();
					v1.normalize();

					// build cross matrix
					cross = v1 ^ v2;
					v2 = cross ^ v1;

					// building the matrix
					mat = MMatrix();
					mat[0][0] = v1.x;
					mat[0][1] = v1.y;
					mat[0][2] = v1.z;
					mat[0][3] = 0;
					mat[1][0] = v2.x;
					mat[1][1] = v2.y;
					mat[1][2] = v2.z;
					mat[1][3] = 0;
					mat[2][0] = cross.x;
					mat[2][1] = cross.y;
					mat[2][2] = cross.z;
					mat[2][3] = 0;
					mat[3][0] = 0;
					mat[3][1] = 0;
					mat[3][2] = 0;
					mat[3][3] = 1;

					// accumulate the newly computed delta
					delta += (dataPoints[i].delta[n] * mat);
				}
			}

			// averaging
			delta /= double(dataPoints[i].size);
			// scaling the vertex
			delta = delta.normal() * (dataPoints[i].deltaLen * applyDeltaV * globalScaleV);
			// adding the delta to the position
			final[i] += delta;

			// now we generate a vector from the final position compute and the starging one
			// and we scale it by the weights and the envelope. "pos" is the original input mesh
			delta = final[i] - pos[i];

			// querying the weight
			weight = weightValue(data, multiIdx, i);
			// FIXME: i = 0,1,2 ‚¾‚¯ weight ‚ªˆÙí‚È’l‚É‚È‚Á‚Ä‚¢‚é
			weight = fmaxf(0, fminf(weight, 1));
			// finally setting the final position
			final[i] = pos[i] + (delta * weight * envelopeV);
		}

		// setting all the points
		iter.setAllPositions(final);
	}

	return returnStat;
}

void CustomDeltaMushDeformer::initData(MObject& mesh, int iters)
{
	// building mfn mesh
	MFnMesh meshFn(mesh);
	// extract the number of vertices
	int size = meshFn.numVertices();

	// scaling the data points array
	dataPoints.resize(size);

	MPointArray pos, res;
	// using a mesh vertex iterator, we need that to extract the neighbours
	MItMeshVertex iter(mesh);
	iter.reset();
	// extracting the world position
	meshFn.getPoints(pos, MSpace::kWorld);

	MVectorArray arr;
	// loop the vertices
	for (int i = 0; i < size; i++, iter.next())
	{
		// creating a new point
		point_data pt;
		// querying the neighbours
		iter.getConnectedVertices(pt.neighbours);
		// extracting neighbour size
		pt.size = pt.neighbours.length();
		// setting the point in the right array
		dataPoints[i] = pt;

		// pre-allocating the deltas array
		arr = MVectorArray();
		arr.setLength(pt.size);
		dataPoints[i].delta = arr;
	}
}

void CustomDeltaMushDeformer::averageRelax(MPointArray& source, MPointArray& target, int iter, double amountV)
{
	// rescaling the target array if needed
	int size = source.length();
	target.setLength(size);

	// making a copy of the original
	MPointArray copy;
	copy.copy(source);

	MVector tmp;
	int i, n, it;
	// looping for how many smooth iterations we have
	for (it = 0; it < iter; it++)
	{
		// looping for the mesh size to smooth the given mesh
		for (i = 0; i < size; i++)
		{
			// resetting the vector
			tmp = MVector::zero;
			// looping the neighbours
			for (n = 0; n < dataPoints[i].size; n++)
			{
				tmp += copy[dataPoints[i].neighbours[n]];
			}

			// perform the average
			tmp /= double(dataPoints[i].size);

			// scaling the final position by the amount the user has set
			target[i] = copy[i] + (tmp - copy[i]) * amountV;
		}
		// copy the target array to be the source of the next iteration
		copy.copy(target);
	}
}

void CustomDeltaMushDeformer::computeDelta(MPointArray& source, MPointArray& target)
{
	int size = source.length();
	MVectorArray arr;
	MVector delta, v1, v2, cross;
	int i, n;
	MMatrix mat;

	// build the matrix
	for (i = 0; i < size; i++)
	{
		delta = MVector(source[i] - target[i]);

		point_data& point = dataPoints[i];
		
		point.deltaLen = delta.length();
		// get tangent matrices
		for (n = 0; n < point.size - 1; n++)
		{
			v1 = target[point.neighbours[n]] - target[i];
			v2 = target[point.neighbours[n + 1]] - target[i];

			v2.normalize();
			v1.normalize();

			cross = v1 ^ v2;
			v2 = cross ^ v1;

			mat = MMatrix();
			mat[0][0] = v1.x;
			mat[0][1] = v1.y;
			mat[0][2] = v1.z;
			mat[0][3] = 0;
			mat[1][0] = v2.x;
			mat[1][1] = v2.y;
			mat[1][2] = v2.z;
			mat[1][3] = 0;
			mat[2][0] = cross.x;
			mat[2][1] = cross.y;
			mat[2][2] = cross.z;
			mat[2][3] = 0;
			mat[3][0] = 0;
			mat[3][1] = 0;
			mat[3][2] = 0;
			mat[3][3] = 1;

			point.delta[n] = MVector(delta * mat.inverse());
		}
	}
}

void CustomDeltaMushDeformer::rebindData(MObject& mesh, int iter, double amount)
{
	// basically resized arrays and backing down neighbours
	initData(mesh, iter);
	MPointArray posRev, back;
	MFnMesh meshFn(mesh);
	meshFn.getPoints(posRev, MSpace::kObject);
	back.copy(posRev);
	// calling the smooth function
	averageRelax(posRev, back, iter, amount);
	// computing the deltas
	computeDelta(posRev, back);
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

	{
		MFnTypedAttribute tAttr;
		MFnNumericAttribute nAttr;

		globalScale = nAttr.create("globalScale", "gls", MFnNumericData::kDouble, 1.0);
		nAttr.setKeyable(true);
		nAttr.setStorable(true);
		nAttr.setMin(0.0001);
		addAttribute(globalScale);

		rebind = nAttr.create("rebind", "rbn", MFnNumericData::kBoolean, 0);
		nAttr.setKeyable(true);
		nAttr.setStorable(true);
		addAttribute(rebind);

		applyDelta = nAttr.create("applyDelta", "apdlt", MFnNumericData::kDouble, 1.0);
		nAttr.setKeyable(true);
		nAttr.setStorable(true);
		nAttr.setMin(0);
		nAttr.setMax(1);
		addAttribute(applyDelta);

		iterations = nAttr.create("iterations", "itr", MFnNumericData::kInt, 0);
		nAttr.setKeyable(true);
		nAttr.setStorable(true);
		nAttr.setMin(0);
		addAttribute(iterations);

		amount = nAttr.create("amount", "am", MFnNumericData::kDouble, 0.5);
		nAttr.setKeyable(true);
		nAttr.setStorable(true);
		nAttr.setMin(0);
		nAttr.setMax(1);
		addAttribute(amount);

		referenceMesh = tAttr.create("referenceMesh", "rfm", MFnData::kMesh);
		tAttr.setKeyable(true);
		tAttr.setWritable(true);
		tAttr.setStorable(true);
		addAttribute(referenceMesh);

		attributeAffects(referenceMesh, outputGeom);
		attributeAffects(rebind, outputGeom);
		attributeAffects(iterations, outputGeom);
		attributeAffects(amount, outputGeom);
		attributeAffects(globalScale, outputGeom);

		//MGlobal::executeCommand("makePaintable -attrType multiFloat -sm deformer CustomDeltaMushDeformer weights");
	}

	return returnStat;
}
