/* 
 * File:   FM.cpp
 * Author: boyko_mihail
 * 
 * Created on 13 октября 2019 г., 18:15
 */

#include "FM.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <map>
#include <cmath>
#include <algorithm>
#include <eigen3/Eigen/Sparse>
#include <iomanip>
#include "RMSE_metric.h"
#include "R2_metric.h"
#include "Statistic.h"

using namespace std;

using namespace Eigen;

FM::FM() {
    this->learning_rate = 0.01;
    this->numEpoh = 5;
    this->bach_size = 1000;
    this->_k = 10000;
    this->maxUsers = 2649420;
    this->maxItem = 17770;
}

FM::FM(double learning_rate, int numEpoh, long int bach_size, int k, long int maxUsers, long int maxItem) {

    this->learning_rate = learning_rate;
    this->numEpoh = numEpoh;
    this->bach_size = bach_size;
    this->_k = k;
    this->maxUsers = maxUsers;
    this->maxItem = maxItem;

}

FM::~FM() {
}

VectorXf FM::predict(const Eigen::SparseMatrix<float, ColMajor> &X_test) {
    cout << "predict big" << endl;
    VectorXf W_0;
    W_0.setConstant(X_test.rows(), this->w0);

    //    Eigen::SparseMatrix<float> firstSlag = X_test*V;
    
    //    Eigen::SparseMatrix<float> firstSlagPowV(firstSlag.rows(), firstSlag.cols());
    //    firstSlagPowV.reserve(firstSlag.nonZeros());
    //    for (Index c = 0; c < firstSlag.cols(); ++c) {
    //        for (Eigen::SparseMatrix<float>::InnerIterator itL(firstSlag, c); itL; ++itL)
    //            firstSlagPowV.insertBack(itL.row(), itL.col()) += itL.value() * itL.value();
    //    }

    MatrixXf powV = this->V.array().pow(2);
    cout << "powV" << endl;
    //    Eigen::SparseMatrix<float, ColMajor> powV(V.rows(), V.cols());
    //    powV.reserve(V.nonZeros());
    //    for (Index c = 0; c < V.cols(); ++c) {
    //        for (Eigen::SparseMatrix<float, ColMajor>::InnerIterator itL(V, c); itL; ++itL)
    //            powV.insertBack(itL.row(), itL.col()) += itL.value() * itL.value();
    //    }

    // Eigen::SparseMatrix<float> secondSlag = ;

    //Eigen::SparseMatrix<float> resultMatrix = firstSlagPowV - secondSlag;
//    MatrixXf first = ();
//    first = first.array().pow(2);
//    cout << "first" << endl;

    MatrixXf resultMatrix = (X_test * V - X_test * powV);
    cout << "resultMatrix" << endl;
    MatrixXf resultVector = resultMatrix.col(0) + resultMatrix.col(1);

    cout << "resultVector" << endl;
    for (int k = 2; k < _k; ++k) {
        resultVector = resultVector + resultMatrix.col(k);
    }
    
    resultVector = resultVector * 0.5;

    return W_0 + X_test * W + resultVector;
}

void FM::fit(const Eigen::SparseMatrix<float, ColMajor> &Xt, const VectorXf &Yt) {

    srand(static_cast<unsigned> (time(0)));

    this->W.setZero(Xt.cols());
    //    this->W.resize(0, 0);
    //    this->W.resize(Xt.cols(), 1);
    //    this->W.reserve(Xt.cols());




    for (long int i = 0; i < Xt.cols(); ++i) {
        int sign = rand() % 2 == 1 ? 1 : -1;
        float abs = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
        W(i) = (sign * abs);
    }


    //    this->V.resize(0, 0);
    //    this->V.resize(Xt.cols() , _k);
    //    this->V.reserve(Xt.cols() * _k);

    //this->V.clear();
    this->V.setZero(Xt.cols(), _k);
    //
    for (long int i = 0; i < Xt.cols(); ++i) {
        for (long int j = 0; j < _k; ++j) {
            int sign = rand() % 2 == 1 ? 1 : -1;
            float abs = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
            V(i, j) = (sign * abs);
        }
    }

    this->w0 = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);


    this->gradientDescent(Xt, Yt);
}

void FM::gradientDescent(const Eigen::SparseMatrix<float, ColMajor> &X, const VectorXf &Y) {

    int k = 0;

    std::vector<int> indexes(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        indexes[i] = i;
    }

    VectorXf lastDiff = VectorXf();
    lastDiff.setZero(bach_size);

    while (k < numEpoh) {

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(indexes.begin(), indexes.end(), g);

        VectorXf newW = this->W;
        MatrixXf newV = this->V;
        float newW0 = this->w0;

        Eigen::SparseMatrix<float, ColMajor> bachX;
        VectorXf bachY;
        for (int i = 0; i < X.rows(); i += this->bach_size) {

            if (indexes[i] + bach_size < X.rows()) {
                bachX = X.block(indexes[i], 0, bach_size, X.cols());
                bachY = Y.block(indexes[i], 0, bach_size, 1);
            } else {
                bachX = X.block(indexes[i], 0, X.rows() - indexes[i], X.cols());
                bachY = Y.block(indexes[i], 0, X.rows() - indexes[i], 1);
            }

            VectorXf current_predict_value = predict_value(newW0, newW, newV, bachX);

            VectorXf diff = (bachY - current_predict_value); // / (((bachY - current_predict_value)*(bachY - current_predict_value)).sqrt());

            diff = diff* learning_rate;
            newW = (newW + (((bachX.transpose() * diff) / bachY.size())));
            W = newW;
            VectorXf ones;
            ones.setOnes(bachY.size());
            newW0 + ((diff.transpose() * ones) / bachY.size())(0, 0);
            newW0 = (newW0 + ((diff.transpose() * ones) / bachY.size())(0, 0));
            w0 = newW0;


            MatrixXf M;
            M.setZero(V.rows(), V.cols());

            for (Index c = 0; c < bachX.cols(); ++c) {
                for (Eigen::SparseMatrix<float>::InnerIterator itL(bachX, c); itL; ++itL)
                    for (int rr = 0; rr < V.cols(); ++rr) {
                        M(c, rr) += V(c, rr) * itL.value() * itL.value();
                    }
            }

            newV = (newV + (((bachX.transpose() * ConstantSumm - M)) / bachY.size()));
            M.resize(0, 0);
            V = newV;
            srand(time(NULL));
            int indexCurrent = 0;
            if (lastDiff.size() == diff.size()) {
                indexCurrent = rand() % diff.size();
            } else if (lastDiff.size() < diff.size()) {
                indexCurrent = rand() % lastDiff.size();
            }

            if (learning_rate > 0.00000000000000000015) {
                if ((lastDiff[indexCurrent] * diff[indexCurrent]) <= 0) {
                    learning_rate *= 0.999;
                    lastDiff = diff;
                } else {
                    learning_rate *= 1.001;
                    lastDiff = diff;
                }
            }
        }
        auto Y_pred = predict(X);
        cout << "predict return" << endl;
        float result_RMSE = RMSE_metric::calculateMetric(Y_pred, Y);
        cout << "learning_rate = " << learning_rate << endl;
        cout << "result RMSE trening epoch #" << k << " = " << result_RMSE << endl;
        ++k;
    }
}

VectorXf FM::predict_value(float W_0, const VectorXf &Wnew, const MatrixXf &Vnew, const Eigen::SparseMatrix<float, ColMajor> &features) {


    ConstantSumm = features*Vnew;

    VectorXf W_0vector;
    W_0vector.setConstant(features.rows(), this->w0);


    //    Eigen::SparseMatrix<float> firstSlag = X_test*V;
    //
    //    Eigen::SparseMatrix<float> firstSlagPowV(firstSlag.rows(), firstSlag.cols());
    //    firstSlagPowV.reserve(firstSlag.nonZeros());
    //    for (Index c = 0; c < firstSlag.cols(); ++c) {
    //        for (Eigen::SparseMatrix<float>::InnerIterator itL(firstSlag, c); itL; ++itL)
    //            firstSlagPowV.insertBack(itL.row(), itL.col()) += itL.value() * itL.value();
    //    }

    MatrixXf powV = this->V.array().pow(2);

    //    Eigen::SparseMatrix<float, ColMajor> powV(V.rows(), V.cols());
    //    powV.reserve(V.nonZeros());
    //    for (Index c = 0; c < V.cols(); ++c) {
    //        for (Eigen::SparseMatrix<float, ColMajor>::InnerIterator itL(V, c); itL; ++itL)
    //            powV.insertBack(itL.row(), itL.col()) += itL.value() * itL.value();
    //    }

    // Eigen::SparseMatrix<float> secondSlag = ;

    //Eigen::SparseMatrix<float> resultMatrix = firstSlagPowV - secondSlag;
    MatrixXf first = (features * V);
    first = first.array().pow(2);

    //    for (Index c = 0; c < first.cols(); ++c) {
    //        for (Eigen::SparseMatrix<float, ColMajor>::InnerIterator itL(first, c); itL; ++itL)
    //            first.insertBack(itL.row(), itL.col()) += itL.value() * itL.value();
    //    }

    MatrixXf resultMatrix = (first - features * powV);
    MatrixXf resultVector = resultMatrix.col(0) + resultMatrix.col(1);
    for (int k = 2; k < _k; ++k) {
        resultVector = resultVector + resultMatrix.col(k);
    }

    resultVector = resultVector * 0.5;


    return W_0vector + features * W + resultVector;
}



