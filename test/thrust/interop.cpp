#include <af/array.h>
#include <gtest/gtest.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust.h>


int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

//thrust tests
//
template<typename T>
class ArrayInterop : public ::testing::Test
{

};

TEST(ArrayInterop, ArrayToThrustDeviceVector)
{
    const int num_elements = 10;
    af::array arr = af::seq(0, num_elements);
    thrust::device_vector<float> t_dvec = toDeviceVector<float>(arr);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(t_dvec.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(t_dvec[i], arr_h[i]);
    }
}

TEST(ArrayInterop, ArrayToThrustHostVector)
{
    const int num_elements = 10;
    af::array arr = af::seq(0, num_elements);
    thrust::host_vector<float> t_hvec = toHostVector<float>(arr);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(t_hvec.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(t_hvec[i], arr_h[i]);
    }
}

TEST(ArrayInterop, ThrustDeviceVectorToArray)
{
    const int num_elements = 10;
    thrust::host_vector<float> t_dvec(num_elements);
    thrust::sequence(t_dvec.begin(), t_dvec.end());

    af::array arr = array(t_dvec);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(t_dvec.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(t_dvec[i], arr_h[i]);
    }
}

TEST(ArrayInterop, ThrustHostVectorToArray)
{
    const int num_elements = 10;
    thrust::host_vector<float> t_hvec(num_elements);
    thrust::sequence(t_hvec.begin(), t_hvec.end());

    af::array arr = array(t_hvec);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(t_hvec.size(), arr.elements());
    for(int i=0; i<num_elements; ++i) {
        ASSERT_EQ(t_hvec[i], arr_h[i]);
    }
}
