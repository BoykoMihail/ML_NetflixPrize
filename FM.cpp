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
    this->crossValScore = 5;
    this->maxUsers = 2649420;
    this->maxItem = 17770;
}

FM::FM(double learning_rate, int numEpoh, int bach_size, int k, int crossValScore, long int maxUsers, long int maxItem) {

    this->learning_rate = learning_rate;
    this->numEpoh = numEpoh;
    this->bach_size = bach_size;
    this->_k = k;
    this->crossValScore = crossValScore;
    this->maxUsers = maxUsers;
    this->maxItem = maxItem;


}

Eigen::SparseMatrix<double> FM::predict(const Eigen::SparseMatrix<double> &X_test) {




    VectorXd W_0;
    W_0.setConstant(X_test.rows(), this->w0);

    double summ = 0;

    Eigen::SparseMatrix<double> firstSlag = X_test*V;

    Eigen::SparseMatrix<double> firstSlagPowV(firstSlag.rows(), firstSlag.cols());
    firstSlagPowV.reserve(firstSlag.nonZeros());
    for (Index c = 0; c < firstSlag.cols(); ++c) {
        for (Eigen::SparseMatrix<double>::InnerIterator itL(firstSlag, c); itL; ++itL)
            firstSlagPowV.insertBack(itL.row(), itL.col()) += itL.value() * itL.value();
    }

    Eigen::SparseMatrix<double> powV(V.rows(), V.cols());
    powV.reserve(V.nonZeros());
    for (Index c = 0; c < V.cols(); ++c) {
        for (Eigen::SparseMatrix<double>::InnerIterator itL(V, c); itL; ++itL)
            powV.insertBack(itL.row(), itL.col()) += itL.value() * itL.value();
    }

    Eigen::SparseMatrix<double> secondSlag = X_test*powV;

    Eigen::SparseMatrix<double> resultMatrix = firstSlagPowV - secondSlag;

    Eigen::SparseMatrix<double> resultVector = resultMatrix.col(0) + resultMatrix.col(1);
    for (int k = 2; k < _k; ++k) {
        resultVector = resultVector + resultMatrix.col(k);
    }

    resultVector = resultVector * 0.5;


    Eigen::SparseMatrix<double> v(W_0 + X_test * W + resultVector);
    return v;
}

void FM::fit(const Eigen::SparseMatrix<double> &Xt, const Eigen::SparseMatrix<double> &Yt) {
    this->X = Xt;

    this->Y = Yt;

    Eigen::SparseMatrix<double> Wn(X.cols(), 1);
    Wn.reserve(X.cols());
    
//    for(long int i = 0; i<X.cols(); ++i){
//        Wn.insert(i,0) = (double) rand() / RAND_MAX;
//    }
    this->W = Wn;


    Eigen::SparseMatrix<double> Vn(X.cols(), _k);
    Vn.reserve(X.cols()*_k);
    
//    for(long int i = 0; i<X.cols(); ++i){
//        for(long int j = 0; j<_k; ++j){
//            Vn.insert(i,j) = (double) rand() / RAND_MAX;
//        }
//    }
    
    this->V = Vn;
    

    this->w0 = (double) rand() / RAND_MAX;


    this->gradientDescent();
}

Eigen::SparseMatrix<double> FM::gradientDescent() {

    int k = 0;

    std::vector<int> indexes(X.rows());
    for (int i = 0; i < X.rows(); ++i) {
        indexes[i] = i;
    }
    
    double lastRMSE = 0;

    while (k < numEpoh) {

        std::random_device rd;
        std::mt19937 g(rd());

        std::shuffle(indexes.begin(), indexes.end(), g);

        Eigen::SparseMatrix<double> newW = this->W;
        Eigen::SparseMatrix<double> newV = this->V;
        double newW0 = this->w0;

        for (int i = 0; i < X.rows(); i += this->bach_size) {

            Eigen::SparseMatrix<double> bachX;
            Eigen::SparseMatrix<double> bachY;
            if (indexes[i] + bach_size < X.rows()) {
                bachX = X.block(indexes[i], 0, bach_size, X.cols());
                bachY = Y.block(indexes[i], 0, bach_size, 1);
            } else {
                bachX = X.block(indexes[i], 0, X.rows() - indexes[i], X.cols());
                bachY = Y.block(indexes[i], 0, X.rows() - indexes[i], 1);
            }

            Eigen::SparseMatrix<double> current_predict_value = predict_value(newW, bachX);
            Eigen::SparseMatrix<double> diff = (bachY - current_predict_value); // / (((bachY - current_predict_value)*(bachY - current_predict_value)).sqrt());

            diff = diff* learning_rate;


           

            newW = (newW + (((bachX.transpose() * diff) / bachY.size())));
            W = newW;

            VectorXd ones;
            ones.setOnes(bachY.size());
            newW0 + ((diff.transpose() * ones)/ bachY.size())(0,0);
            newW0 = (newW0 + ((diff.transpose() * ones)/ bachY.size())(0,0));
            w0 = newW0;


            Eigen::SparseMatrix<double> M(V.rows(), V.cols());
            M.reserve(V.nonZeros());
            for (int r = 0; r < bachX.rows(); ++r) {
                for (Index c = 0; c < V.cols(); ++c) {
                    M.startVec(c);
                    for (Eigen::SparseMatrix<double>::InnerIterator itL(V, c); itL; ++itL)
                        M.insertBack(itL.row(), c) += itL.value() * bachX.insert(r, c);
                }
            }
            M.finalize();
            newV = (newV + (((bachX.transpose() * ConstantSumm - M)) / bachY.size()));

            V = newV;

        }
        auto Y_pred = predict(X);
        double result_RMSE = RMSE_metric::calculateMetric(Y_pred, Y);

        cout << "result RMSE trening epoch #" << k << " = " << result_RMSE << endl;
        if (lastRMSE > result_RMSE) {
            learning_rate *= 1.000001;
        } else if(lastRMSE < result_RMSE) {
            learning_rate *= 0.999999;
        }
        ++k;
    }
    return W;
}

Eigen::SparseMatrix<double> FM::predict_value(const Eigen::SparseMatrix<double> &ntheta, const Eigen::SparseMatrix<double> &features) {

    ConstantSumm = features*V;
    return features * ntheta;
}



