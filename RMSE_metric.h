/* 
 * File:   RMSE_metric.h
 * Author: boyko_mihail
 *
 * Created on 24 сентября 2019 г., 12:28
 */
#include "Metric.h"

#ifndef RMSE_METRIC_H
#define	RMSE_METRIC_H

class RMSE_metric : public Metric {
public:

    static double calculateMetric(const VectorXf &Y_pred, const VectorXf &Y_test) {

        double sum = 0;

        sum += (Y_test - Y_pred).dot(Y_test - Y_pred);

        return sqrt(sum / Y_pred.rows());
    }
};

#endif	/* RMSE_METRIC_H */

