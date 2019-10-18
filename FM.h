/* 
 * File:   FM.h
 * Author: boyko_mihail
 *
 * Created on 13 октября 2019 г., 18:15
 */

#ifndef FM_H
#define	FM_H

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
//#include <eigen/Sparse>
#include <eigen3/Eigen/Sparse>

using namespace Eigen;
using namespace std;

class FM {
public:
    FM();
    FM(double learning_rate, int numEpoh, long int bach_size, int k, long int maxUsers, long int maxItem);

    void fit(const Eigen::SparseMatrix<float, ColMajor> &Xt, const VectorXf &Yt);

    VectorXf predict(const Eigen::SparseMatrix<float, ColMajor> &X_test);

    std::vector<float> getW() {
        std::vector<float> w(0);
        for (int i = 0; i < W.size(); ++i) {
            w.push_back(W(i, 0));
        }
        return w;
    }

    ~FM();

private:
    MatrixXf V;
    VectorXf W;
    MatrixXf ConstantSumm;
    float w0, learning_rate;
    int numEpoh, _k;
    long int bach_size;
    long int maxUsers, maxItem;



    VectorXf predict_value(float W_0, const VectorXf &Wnew, const MatrixXf &Vnew, const Eigen::SparseMatrix<float, ColMajor> &features);
    void gradientDescent(const Eigen::SparseMatrix<float, ColMajor> &X, const VectorXf &Y);


};

#endif	/* FM_H */

