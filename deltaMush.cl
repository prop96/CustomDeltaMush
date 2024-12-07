__constant float FOUR_INV = 0.25f;

__inline__ float3 mat3_vec3_mul(
        const float3 mx,
        const float3 my,
        const float3 mz,
        const float3 v)
{
    float3 res;
    res.x = (mx.x * v.x) + (my.x * v.y) +(mz.x * v.z);
    res.y = (mx.y * v.x) + (my.y * v.y) +(mz.y * v.z);
    res.z = (mx.z * v.x) + (my.z * v.y) +(mz.z * v.z);
    return res;
}

__inline__ float float3_len(const float3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__inline__ float3 vec_norm(float3 v)
{
    float len = 1.0f / float3_len(v);
    float3 res;
    res.x = v.x * len;
    res.y = v.y * len;
    res.z = v.z * len;
    return res;
}

__kernel void AverageOpencl(
    __global float* finalPos ,
    __global uint* startIdxTable,
    __global int* neig_table,
    __global const float* initialPos ,
    const float amount,
    const uint numPoints
    )
{
    unsigned int pointIdx = get_global_id(0);
    if ( pointIdx >= numPoints )
    {
        return;
    }
    
    const float3 pos = vload3( pointIdx , initialPos );

    const uint startIdx = startIdxTable[pointIdx];
    const uint numNeighbours = startIdxTable[pointIdx + 1] - startIdx

    float3 v = {0.0f, 0.0f, 0.0f};
    int id;
    for (int i=0; i < numNeighbours; i++)
    {
        id = neig_table[startIdx + i] * 3;
        v.x += initialPos[id];
        v.y += initialPos[id+1];
        v.z += initialPos[id+2];
    }

    const float numNeighboursInv = 1.0f / numNeighbours;

    v.x *= numNeighboursInv;
    v.y *= numNeighboursInv;
    v.z *= numNeighboursInv;

    v = pos + ((v - pos) * amount);
    
    vstore3( v , pointIdx , finalPos );
}


__kernel void TangentSpaceOpencl(
    __global float* finalPos,
    __global uint* startIdxTable,
    __global float* d_delta_table,
    __global float* d_delta_size,
    __global int* d_neig_table,
    __global const float* initialPos ,
    const uint numPoints
    )
{
    const unsigned int pointIdx = get_global_id(0);
    if ( pointIdx >= numPoints )
    {
        return;
    }

    float3 accum = {0.0f, 0.0f, 0.0f};
    float3 v0, v1, v2, crossV, delta, deltaRef;
    unsigned int id;
    v0 = vload3(pointIdx, initialPos);

    for (unsigned int i = 0; i < 3; i++)
    {
        id = d_neig_table[pointIdx * 4 + i];
        v1 = vload3(id, initialiPos);

        id = d_neig_table[pointIdx * 4 + i + 1];
        v2 = vload3(id, initialiPos);

        v1 -= v0;
        v2 -= v0;
        v1 = normalize(v1);
        v2 = normalize(v2);

        crossV = cross(v1, v2);
        v2 = cross(crossV, v2);

        id = pointIdx * 9 + i * 3;
        deltaRef.x = d_delta_table[id];
        deltaRef.y = d_delta_table[id + 1];
        deltaRef.z = d_delta_table[id + 2];

        delta = mat3_vec3_mult(v1, v2, crossV, deltaRef);
        accum += delta;
    }
    accum = vec_norm(accum);
    accum *= d_delta_size[pointIdx];
    accum += v0;
    vstore3(accum, pointIdx, finalPos);
}