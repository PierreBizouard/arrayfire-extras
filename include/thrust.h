/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#include <af/array.h>
#include <af/defines.h>
#include <af/traits.hpp>
#include <af/exception.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


/**
   Interop function that converts a thrust::device_vector to an af::array

   \param[in] tvec thrust::device_vector to be converted
   \return af::array with converted contents of the thrust vector

   \ingroup cuda_interop
 */
template<typename T>
af::array array(thrust::device_vector<T> &tvec)
{
    af::array output;
    if(tvec.empty()) return output;
    T* ptr = thrust::raw_pointer_cast(tvec.data());
    af::array tmp = af::array(tvec.size(), ptr, afDevice);
    output = tmp.copy();
    tmp.lock();
    return output;
}

/**
   Interop function that converts a thrust::host_vector to an af::array

   \param[in] tvec thrust::host_vector to  be converted
   \return af::array with converted contents of the thrust vector

   \ingroup cuda_interop
 */
template<typename T>
af::array array(thrust::host_vector<T> &tvec)
{
    af::array output;
    if(tvec.empty()) return output;
    //af_dtype type = (af_dtype)dtype_traits<T>::af_type;
    T* ptr = thrust::raw_pointer_cast(tvec.data());
    output = af::array(tvec.size(), ptr);
    return output;
}

/**
   Interop function that converts an af::array to a thrust::host_vector

   \param[in] arr af::array to be converted to thrust::host_vector
   \return thrust::host_vector with converted contents of af::array

   \ingroup cuda_interop
 */
template<typename T>
thrust::host_vector<T> toHostVector(af::array &arr)
{
    af_dtype vec_type = (af_dtype)af::dtype_traits<T>::af_type;
    if(arr.type() != vec_type)
        throw af::exception("Thrust vector data type and array data type are mismatching");
    thrust::host_vector<T> hvec(arr.elements());
    T* h_ptr = thrust::raw_pointer_cast(hvec.data());
    arr.host(h_ptr);
    return hvec;
}

/**
   Interop function that converts an af::array to a thrust::device_vector

   \param[in] arr af::array to be converted to thrust::device_vector
   \return thrust::device_vector with converted contents of af::array

   \ingroup cuda_interop
 */
template<typename T>
thrust::device_vector<T> toDeviceVector(af::array &arr)
{
    af_dtype vec_type = (af_dtype)af::dtype_traits<T>::af_type;
    if(arr.type() != vec_type)
        throw af::exception("Thrust vector data type and array data type are mismatching");
    arr.lock();
    thrust::device_ptr<T> af_ptr = thrust::device_pointer_cast(arr.device<T>());
    thrust::device_vector<T> dvec(af_ptr, af_ptr + arr.elements());
    arr.unlock();
    return dvec;
}
