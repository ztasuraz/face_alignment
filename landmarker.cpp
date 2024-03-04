#ifndef FACE_DEMO_FACEPREPROCESS_H
#define FACE_DEMO_FACEPREPROCESS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cpptrace/cpptrace.hpp>
#include <numeric>
// #include "data_class.h"
const auto raw_trace = cpptrace::generate_raw_trace();

namespace FacePreprocess
{
    const cv::Mat default_dst_landmark = (cv::Mat_<float>(5, 2) << 38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366, 41.5493, 92.3655, 70.7299, 92.2041);

    cv::Mat meanAxis0(const cv::Mat &src)
    {
        int num = src.rows;
        int dim = src.cols;

        // x1 y1
        // x2 y2

        cv::Mat output(1, dim, CV_32F);
        for (int i = 0; i < dim; i++)
        {
            float sum = 0;
            for (int j = 0; j < num; j++)
            {
                sum += src.at<float>(j, i);
            }
            output.at<float>(0, i) = sum / num;
        }

        return output;
    }

    cv::Mat elementwiseMinus(const cv::Mat &A, const cv::Mat &B)
    {
        cv::Mat output(A.rows, A.cols, A.type());

        assert(B.cols == A.cols);
        if (B.cols == A.cols)
        {
            for (int i = 0; i < A.rows; i++)
            {
                for (int j = 0; j < B.cols; j++)
                {
                    output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
                }
            }
        }
        return output;
    }

    cv::Mat varAxis0(const cv::Mat &src)
    {
        cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
        cv::multiply(temp_, temp_, temp_);
        return meanAxis0(temp_);
    }

    int MatrixRank(cv::Mat M)
    {
        cv::Mat w, u, vt;
        cv::SVD::compute(M, w, u, vt);
        cv::Mat1b nonZeroSingularValues = w > 0.0001;
        int rank = countNonZero(nonZeroSingularValues);
        return rank;
    }

    cv::Mat similarTransform(cv::Mat src, cv::Mat dst)
    {
        int num = src.rows;
        int dim = src.cols;
        cv::Mat src_mean = meanAxis0(src);
        cv::Mat dst_mean = meanAxis0(dst);
        cv::Mat src_demean = elementwiseMinus(src, src_mean);
        cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
        cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
        cv::Mat d(dim, 1, CV_32F);
        // std::cout << "A = " << A << std::endl;
        // print matrix rows and columns, type
        // std::cout << "A rows: " << A.rows << " A cols: " << A.cols << " A type: " << A.type() << " CV32F " << CV_32F << " CV64F " << CV_64F << std::endl;
        d.setTo(1.0f);
        if (cv::determinant(A) < 0)
        {
            d.at<float>(dim - 1, 0) = -1;
        }
        // std::cout << "d = " << d << std::endl;
        cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
        // std::cout << "T = " << T << std::endl;
        cv::Mat U, S, V;
        cv::SVD::compute(A, S, U, V);
        // std::cout << "U = " << U << std::endl;
        int rank = MatrixRank(A);
        if (rank == 0)
        {
            assert(rank == 0);
        }
        else if (rank == dim - 1)
        {
            if (cv::determinant(U) * cv::determinant(V) > 0)
            {
                T.rowRange(0, dim).colRange(0, dim) = U * V;
            }
            else
            {
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;

                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_ * V; // np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U * twp;
                d.at<float>(dim - 1, 0) = s;
            }
        }
        else
        {
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V.t(); // np.dot(np.diag(d), V.T)
            cv::Mat res = U * twp;       // U
            T.rowRange(0, dim).colRange(0, dim) = U * diag_ * V;
        }
        cv::Mat var_ = varAxis0(src_demean);
        float val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d, S, res);
        float scale = 1.0 / val * cv::sum(res).val[0];
        cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim) * src_mean.t();
        cv::Mat temp2 = scale * temp1;
        cv::Mat temp3 = dst_mean - temp2.t();
        T.at<float>(0, 2) = temp3.at<float>(0);
        T.at<float>(1, 2) = temp3.at<float>(1);
        T.rowRange(0, dim).colRange(0, dim) *= scale; // T[:dim, :dim] *= scale

        return T;
    }

}
#endif // FACE_DEMO_FACEPREPROCESS_H

// sample data class CGPoint, has x and y properties
class CGPoint
{
public:
    float x;
    float y;
    CGPoint(float x, float y) : x(x), y(y) {}
    void rescale(float width, float height)
    {
        this->x *= width;
        this->y *= height;
    }
};

class CGSize
{
public:
    float width;
    float height;
    CGSize(float width, float height) : width(width), height(height) {}
    void rescale(float width, float height)
    {
        this->width *= width;
        this->height *= height;
    }
};

class CGRect
{
public:
    CGPoint origin;
    CGSize size;
    CGRect(CGPoint origin, CGSize size) : origin(origin), size(size) {}
    void rescale(float width, float height)
    {
        this->origin.rescale(width, height);
        this->size.rescale(width, height);
    }
};

// function to convert a list of CGPoint to cv::Mat
cv::Mat CGPoint2Mat(std::vector<CGPoint> points, int width, int height, CGRect boundingBox)
{
    cv::Mat mat(points.size(), 2, CV_32F);
    // boundingBox.rescale(width, height);
    boundingBox.origin.x = boundingBox.origin.x * width;
    boundingBox.origin.y = boundingBox.origin.y * height;
    boundingBox.size.width = boundingBox.size.width * width;
    boundingBox.size.height = boundingBox.size.height * height;
    double inner_w = boundingBox.size.width;
    double inner_h = boundingBox.size.height;

    for (int i = 0; i < points.size(); i++)
    {
        // rescale the points to inner_w and inner_h
        // points[i].rescale(inner_w, inner_h);
        points[i].x = points[i].x * inner_w;
        points[i].y = points[i].y * inner_h;
        mat.at<float>(i, 0) = points[i].x + boundingBox.origin.x;
        mat.at<float>(i, 1) = points[i].y + boundingBox.origin.y;
    }
    return mat;
}

// main function to align the face, given the image and the landmarks
// output the aligned face image

cv::Mat align_face(cv::Mat image, std::vector<CGPoint> landmarks, CGRect boundingBox)
{
    // get only 13th, 6th, 49th, 34th, 26th from landmarks
    std::vector<CGPoint> five_lmnks = {landmarks[13], landmarks[6], landmarks[49], landmarks[34], landmarks[26]};
    // OPTIONAL: use cv.resize to resize 0.5x
    cv::resize(image, image, cv::Size(), 0.5, 0.5);
    int width = image.cols;
    int height = image.rows;
    // convert the vector of CGPoint to cv::Mat
    cv::Mat src = CGPoint2Mat(five_lmnks, width, height, boundingBox);
    // // loop through src and draw
    // for (int i = 0; i < src.rows; i++)
    // {
    //     cv::circle(image, cv::Point(src.at<float>(i, 0), src.at<float>(i, 1)), 5, cv::Scalar(255, 255, 255), -1);
    // }
    // cv::imwrite("landmarked.jpg", image);
    // cv::imshow("Imaxge", image);
    // cv::waitKey(0);
    // std::cout << "src = " << src << std::endl;
    // LOAD default dst landmark points
    cv::Mat dst_landmark = FacePreprocess::default_dst_landmark;
    cv::Mat warped = cv::Mat::zeros(112, 112, image.type());
    cv::Mat T = FacePreprocess::similarTransform(src, dst_landmark);
    cv::warpAffine(image, warped, T.rowRange(0, 2), warped.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    return warped;
};
int main(int argc, char **argv)
{
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    // extract the image path
    try
    {
        std::string image_path = argv[1];
        std::cout << "Image path: " << image_path << std::endl;
        // sample landmark points
        // first create 76 CGPoint objects
        std::vector<CGPoint> points;
        for (int i = 0; i < 76; i++)
        {
            points.push_back(CGPoint(0, 0));
        }

        // replace 13th, 6th, 49th, 34th, 26th with values above
        points[13] = CGPoint(0.3226475119590759, 0.2660534977912903);
        points[6] = CGPoint(0.696449875831604, 0.2604473829269409);
        points[49] = CGPoint(0.5035454630851746, 0.4475863575935364);
        points[34] = CGPoint(0.3754761219024658, 0.7001306116580963);
        points[26] = CGPoint(0.6636321246623993, 0.6974610090255737);

        CGRect boundingBox = CGRect(CGPoint(0.24391072988510132, 0.33889517188072205), CGSize(0.5635068416595459, 0.31697261333465576));

        // read the image
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty())
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        // call the align_face function
        cv::Mat warped = align_face(image, points, boundingBox);
        cv::imshow("Warped", warped);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (std::exception &e)
    {
        // trace();
        raw_trace.resolve().print();
        std::cerr << "Error: " << e.what() << '\n';
    }
    return 0;
}