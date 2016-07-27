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
#include <vector>
#include <valarray>


/**
   Interop function that converts a std::vector to an af::array

   \param[in] vec std::vector to be converted
   \return af::array with converted contents of the std::vector

   \ingroup stl_interop
 */
template<typename T>
af::array array(const std::vector<T> &vec)
{
    af_array temp = 0;
    dim_t sz = vec.size();
    af_dtype type = (af_dtype)af::dtype_traits<T>::af_type;
    af_err err = af_create_array(&temp, (const void*)&vec[0], 1, &sz, type);
    if(err != AF_SUCCESS)
        throw af::exception("Error creating af::array",  __LINE__, err);
    return af::array(temp);
}

/**
   Interop function that converts a std::valarray to an af::array

   \param[in] varr std::valarray to be converted
   \return af::array with converted contents of the std::valarray

   \ingroup stl_interop
 */
template<typename T>
af::array array(std::valarray<T> &varr)
{
    af_array temp = 0;
    dim_t sz = varr.size();
    af_dtype type = (af_dtype)af::dtype_traits<T>::af_type;
    af_err err = af_create_array(&temp, (const void*)&varr[0], 1, &sz, type);
    if(err != AF_SUCCESS)
        throw af::exception("Error creating af::array",  __LINE__, err);
    return af::array(temp);
}

/**
   Interop function that converts an af::array to an stl::vector

   \param[in] arr af::array to be converted to stl::vector
   \return stl::vector with converted contents of af::array

   \ingroup stl_interop
 */
template<typename T>
std::vector<T> toStdVector(af::array &arr)
{
    std::vector<T> out(arr.elements());
    af::dtype oType = (af::dtype)af::dtype_traits<T>::af_type;
    if(arr.type() != oType) {
        arr.as(oType).host(&out[0]);
    } else {
        arr.host(&out[0]);
    }
    return out;
}

/**
   Interop function that converts an af::array to an stl::valarray

   \param[in] arr af::array to be converted to stl::valarray
   \return stl::valarray with converted contents of af::array

   \ingroup stl_interop
 */
template<typename T>
std::valarray<T> toStdValarray(af::array &arr)
{
    std::valarray<T> out(arr.elements());
    af::dtype oType = (af::dtype)af::dtype_traits<T>::af_type;
    if(arr.type() != oType) {
        arr.as(oType).host(&out[0]);
    } else {
        arr.host(&out[0]);
    }
    return out;
}
