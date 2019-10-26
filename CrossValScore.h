/* 
 * File:   CrossValScore.h
 * Author: boyko_mihail
 *
 * Created on 13 октября 2019 г., 21:26
 */

#ifndef CROSSVALSCORE_H
#define	CROSSVALSCORE_H

#include "FM.h"

class CrossValScore {
public:
    CrossValScore(float learning_rate, int numEpoh, long int bach_size, int k, int countOfCrossVal, long int maxUsers, long int maxItem);


    void fit(Eigen::SparseMatrix<float, RowMajor> &X, VectorXf &Y);

    float getMeanRMSE();
private:

    float meanRMSE;
    long int maxUsers, maxItem;
    int countOfCrossVal;
    float w0, learning_rate;
    int numEpoh, bach_size, _k;



};

#endif	/* CROSSVALSCORE_H */

