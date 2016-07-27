#include <af/array.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <valarray>
#include <stl.h>

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

//stl tests
//
template<typename T>
class ArrayInterop : public ::testing::Test
{

};

TEST(ArrayInterop, ArrayToSTLVector)
{
    const int num_elements = 10;
    af::array arr = af::seq(0, num_elements);
    std::vector<float> vec = toStdVector<float>(arr);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(vec.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(vec[i], arr_h[i]);
    }
}

TEST(ArrayInterop, ArrayToSTLValarray)
{
    const int num_elements = 10;
    af::array arr = af::seq(0, num_elements);
    std::valarray<float> varr = toStdValarray<float>(arr);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(varr.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(varr[i], arr_h[i]);
    }
}

TEST(ArrayInterop, STLVectorToArray)
{
    const int num_elements = 10;
    std::vector<float> vec(num_elements);
    for(int i=0; i<num_elements; ++i) {
        vec[i] = i;
    }

    af::array arr = array(vec);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(vec.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(vec[i], arr_h[i]);
    }
}

TEST(ArrayInterop, STLValarrayToArray)
{
    const int num_elements = 10;
    std::valarray<float> varr(num_elements);
    for(int i=0; i<num_elements; ++i) {
        varr[i] = i;
    }

    af::array arr = array(varr);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(varr.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(varr[i], arr_h[i]);
    }
}
