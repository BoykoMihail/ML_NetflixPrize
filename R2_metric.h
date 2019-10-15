/* 
 * File:   R2_metric.h
 * Author: boyko_mihail
 *
 * Created on 24 сентября 2019 г., 12:28
 */
#include "Metric.h"
#include "Statistic.h"
#include <eigen3/Eigen/Core>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#ifndef R2_METRIC_H
#define	R2_METRIC_H

class R2_metric : public Metric {
public:

    static double calculateMetric(const std::vector<double> Y_pred, const std::vector<double> Y_test) {

        double sum_Up = 0;
        double sum_down = 0;
        double Y_mean = 0;
        double Y_sig = 0;

        VectorXd Y_test_vector = VectorXd::Map(Y_test.data(), Y_test.size());
        VectorXd Y_pred_vector = VectorXd::Map(Y_pred.data(), Y_pred.size());


        Statistic::findeStatistic(Y_test_vector, Y_mean, Y_sig);

        VectorXd one(Y_test_vector.size());
        one.setConstant(Y_mean);
        
        sum_Up += (Y_test_vector - Y_pred_vector).dot((Y_test_vector - Y_pred_vector)); //( Y_test[i] - Y_pred[i])*( Y_test[i] - Y_pred[i]);
        sum_down += (Y_test_vector - one).dot(Y_test_vector - one); //( Y_test[i] - Y_mean)*( Y_test[i] - Y_mean);
        return 1 - (sum_Up / sum_down);
    }
};

#endif	/* R2_METRIC_H */

