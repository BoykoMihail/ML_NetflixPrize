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
    FM(double learning_rate, int numEpoh, int bach_size, int k, int crossValScore, long int maxUsers, long int maxItem);

    void fit(const Eigen::SparseMatrix<double> &Xt, const Eigen::SparseMatrix<double> &Yt);
    
    Eigen::SparseMatrix<double> predict(const Eigen::SparseMatrix<double> &X_test);

    std::vector<double> getW() {
        std::vector<double> w(0);
        for (int i = 0; i < W.size(); ++i) {
            w.push_back(W.insert(i,0));
        }
        return w;
    }

private:
    Eigen::SparseMatrix<double> X, V;
    Eigen::SparseMatrix<double> Y, W;
    Eigen::SparseMatrix<double>  ConstantSumm;
    double w0, learning_rate;
    int numEpoh, bach_size,  _k, crossValScore;
    long int maxUsers, maxItem;



    Eigen::SparseMatrix<double> predict_value(const Eigen::SparseMatrix<double> &ntheta, const Eigen::SparseMatrix<double> &features);
    Eigen::SparseMatrix<double> gradientDescent();


};

#endif	/* FM_H */

