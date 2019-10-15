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
    CrossValScore(FM& model, int countOfCrossVal, long int maxUsers, long int maxItem);

    
    void fit(Eigen::SparseMatrix<double> &X, VectorXd &Y);
    
    double getMeanRMSE();
private:
    
    FM model;
    double meanRMSE;
    long int maxUsers, maxItem;
    int countOfCrossVal;
    
    

};

#endif	/* CROSSVALSCORE_H */

