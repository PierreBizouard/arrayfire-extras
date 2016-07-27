#include <arrayfire.h>
#include <gtest/gtest.h>
#include <iostream>
using namespace std;

#include "opencv2/core/version.hpp"
#if CV_MAJOR_VERSION == 2
#include <opencv.h>
#elif CV_MAJOR_VERSION == 3
#include <opencv3.h>
#endif

float testdata[] = { 1,2,3,4,5,6,
                     1,2,3,4,5,6,
                     1,2,3,4,5,6,
                     1,2,3,4,5,6,
                     1,2,3,4,5,6,
                     1,2,3,4,5,6,
                     1,2,3,4,5,6,
                     1,2,3,4,5,6 };

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

template<typename T>
class ArrayInterop : public ::testing::Test
{

};

// common opencv tests (between versions)
//

TEST(ArrayInterop, CVMatToArraySingleChannel)
{
    const int rows = 8;
    const int cols = 6;
    cv::Mat cvMat (rows, cols, CV_32FC1);
    cvMat.setTo(0);
    cvMat.row(0) = 1;

    af::array arr  = afcv::array(cvMat, false);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols, arr.elements());
}

TEST(ArrayInterop, CVMatToArrayDoubleChannel)
{
    const int rows = 8;
    const int cols = 6;
    const int channels = 2;
    cv::Mat cvMat (rows, cols, CV_32FC2);

    cvMat.setTo(0);
    cvMat.row(0) = 1;
    cvMat.row(1) = 2;

    af::array arr  = afcv::array(cvMat, false);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols * channels, arr.elements());
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            for(int k=0; k<channels; ++k) {
                ASSERT_EQ(cvMat.at<cv::Vec2f>(j, i)[k], arr_h[(rows*cols)*k + cols * j + i]);
            }
        }
    }
}

TEST(ArrayInterop, CVMatToArrayTripleChannel)
{
    const int rows = 8;
    const int cols = 6;
    const int channels = 3;
    cv::Mat cvMat(rows, cols, CV_32FC3);

    cvMat.setTo(0);
    cvMat.row(0) = 1;
    cvMat.row(1) = 2;
    cvMat.row(2) = 3;

    af::array arr  = afcv::array(cvMat, false);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols * channels, arr.elements());
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            for(int k=0; k<channels; ++k) {
                ASSERT_EQ(cvMat.at<cv::Vec3f>(j, i)[k], arr_h[(rows*cols)*k + cols * j + i]);
            }
        }
    }
}

TEST(ArrayInterop, CVMatToArrayQuadChannel)
{
    const int rows = 8;
    const int cols = 6;
    const int channels = 4;
    cv::Mat cvMat(rows, cols, CV_32FC4);

    cvMat.setTo(0);
    cvMat.row(0) = 1;
    cvMat.row(0) = 1;
    cvMat.row(1) = 2;
    cvMat.row(2) = 3;
    cvMat.row(3) = 4;

    af::array arr  = afcv::array(cvMat, false);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols * channels, arr.elements());
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            for(int k=0; k<channels; ++k) {
                ASSERT_EQ(cvMat.at<cv::Vec4f>(j, i)[k], arr_h[(rows*cols)*k + cols * j + i]);
            }
        }
    }
}

TEST(ArrayInterop, CVMatToArraySingleChannelTranspose)
{
    const int rows = 8;
    const int cols = 6;
    cv::Mat cvMat (rows, cols, CV_32FC1);
    cvMat.setTo(0);
    cvMat.row(0) = 1;

    af::array arr = afcv::array(cvMat, false);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols, arr.elements());
    for(int i=0; i<cols; ++i) {
        for(int j=0; j<rows; ++j) {
            ASSERT_EQ(cvMat.at<float>(i, j), arr_h[rows * i + j]);
        }
    }

}

TEST(ArrayInterop, CVMatToArrayQuadChannelTranspose)
{
    const int rows = 8;
    const int cols = 6;
    const int channels = 4;
    cv::Mat cvMat(rows, cols, CV_32FC4);

    cvMat.setTo(0);
    cvMat.row(0) = 1;
    cvMat.row(0) = 1;
    cvMat.row(1) = 2;
    cvMat.row(2) = 3;
    cvMat.row(3) = 4;

    af::array arr  = afcv::array(cvMat, true);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows, arr.dims(0));
    ASSERT_EQ(cols, arr.dims(1));
    ASSERT_EQ(rows * cols * channels, arr.elements());
    for(int i=0; i<cols; ++i) {
        for(int j=0; j<rows; ++j) {
            for(int k=0; k<channels; ++k) {
                ASSERT_EQ(cvMat.at<cv::Vec4f>(j, i)[k], arr_h[(rows*cols)*k + rows * i + j]);
            }
        }
    }
}

TEST(ArrayInterop, CVMatVecToArray)
{
    const int rows = 8;
    const int cols = 6;

    cv::Mat cvMat (8, 6, CV_32FC1);
    cvMat.setTo(0);
    cvMat.row(0) = 1;

    std::vector<cv::Mat> mats;

    mats.push_back(cvMat);
    mats.push_back(cvMat);
    mats.push_back(cvMat);
    mats.push_back(cvMat);

    af::array arr = afcv::array(mats,  false);
    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols * mats.size(), arr.elements());
    for(int m=0; m<mats.size(); ++m) {
        for(int i=0; i<rows; ++i) {
            for(int j=0; j<cols; ++j) {
                ASSERT_EQ(mats[m].at<float>(j, i), arr_h[(rows*cols)*m + rows * j + i]);
            }
        }
    }
}

TEST(ArrayInterop, CVMatVec1DToArray)
{
    const int rows = 8;
    cv::Mat cvMat1D (rows, 1, CV_32FC1);
    cvMat1D.setTo(4);
    cv::Mat cvMat1D_ (rows, 1, CV_32FC1);
    cvMat1D_.setTo(5);
    std::vector<cv::Mat> mats;
    mats.push_back(cvMat1D);
    mats.push_back(cvMat1D_);
    mats.push_back(cvMat1D);
    mats.push_back(cvMat1D_);

    af::array arr = afcv::array(mats);
    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * mats.size(), arr.elements());
    for(int m=0; m<mats.size(); ++m) {
        for(int i=0; i<rows; ++i) {
            ASSERT_EQ(mats[m].at<float>(0, i), arr_h[(rows)*m + i]);
        }
    }
}

TEST(ArrayInterop, CVMatVecInconsistentToArray)
{
    cv::Mat cvMat (8, 6, CV_32FC1);
    cv::Mat cvMat1 (8, 6, CV_32FC2);
    std::vector<cv::Mat> matsmix;

    matsmix.push_back(cvMat);
    matsmix.push_back(cvMat1);
    EXPECT_THROW(afcv::array(matsmix, false), af::exception);
}

TEST(ArrayInterop, ArrayToCvMat)
{
    const int rows = 6;
    const int cols = 8;
    af::array arr(rows, cols, testdata);

    cv::Mat mat = afcv::toMat(arr, CV_32FC1, false);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols, arr.elements());
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            ASSERT_EQ(mat.at<float>(i, j), arr_h[rows * j + i]);
        }
    }
}

TEST(ArrayInterop, ArrayToCvMatTranspose)
{
    const int rows = 6;
    const int cols = 8;
    af::array arr(rows, cols, testdata);

    cv::Mat mat = afcv::toMat(arr, CV_32FC1, true);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols, arr.elements());
    for(int i=0; i<cols; ++i) {
        for(int j=0; j<rows; ++j) {
            ASSERT_EQ(mat.at<float>(i, j), arr_h[rows * i + j]);
        }
    }
}

TEST(ArrayInterop, ArrayQuadChannelToCvMat)
{
    const int rows = 6;
    const int cols = 8;
    const int channels = 4;
    af::array arr(rows, cols, testdata);
    arr = tile(arr, 1, 1, 4);

    cv::Mat mat = afcv::toMat(arr, CV_32F, false);

    float *arr_h = arr.host<float>();
    ASSERT_EQ(rows * cols * channels, arr.elements());
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            for(int k=0; k<channels; ++k) {
                ASSERT_EQ(mat.at<cv::Vec4f>(i, j)[k], arr_h[(rows*cols)*k + rows * j + i]);
            }
        }
    }
}

TEST(ArrayInterop, ArrayToCvMatTypes)
{
    const int rows = 6;
    const int cols = 8;
    const int channels = 4;
    af::array arr(rows, cols, testdata);

    cv::Mat matAfCvF  = afcv::toMat(arr, CV_32F, false);
    ASSERT_EQ(matAfCvF.depth(), CV_32F);
    cv::Mat matAfCvS  = afcv::toMat(arr, CV_32S, true);
    ASSERT_EQ(matAfCvS.depth(), CV_32S);
    cv::Mat matAfCvFF = afcv::toMat(arr, CV_64F, false);
    ASSERT_EQ(matAfCvFF.depth(), CV_64F);
    cv::Mat matAfCvU  = afcv::toMat(arr, CV_8U, false);
    ASSERT_EQ(matAfCvU.depth(), CV_8U);

    EXPECT_THROW(afcv::toMat(arr, CV_32SC4, false), af::exception);
}
