/* 
 * File:   CrossValScore.cpp
 * Author: boyko_mihail
 * 
 * Created on 13 октября 2019 г., 21:26
 */

#include "CrossValScore.h"
#include "RMSE_metric.h"
#include "R2_metric.h"
#include "Statistic.h"
#include "FM.h"

CrossValScore::CrossValScore(float learning_rate, int numEpoh, long int bach_size, int k, int countOfCrossVal, long int maxUsers, long int maxItem) {

    this->countOfCrossVal = countOfCrossVal;
    this->maxUsers = maxUsers;
    this->maxItem = maxItem;
    this->bach_size = bach_size;
    this->learning_rate = learning_rate;
    this->numEpoh = numEpoh;
    this->_k = k;



}

void CrossValScore::fit(Eigen::SparseMatrix<float, ColMajor> &X, VectorXf &Y) {

    int crossValCount = X.rows() / countOfCrossVal;

    std::vector<float> RMSE_results(0);
    std::vector<std::vector<float>> all_W(0);

    Eigen::SparseMatrix<float, ColMajor> X_train;
    Eigen::SparseMatrix<float, ColMajor> X_test;

    Eigen::SparseMatrix<float, ColMajor> X_train2;

    Eigen::SparseMatrix<float, ColMajor> X_train3;

    VectorXf Y_train(X.rows() - crossValCount);
    VectorXf Y_test(crossValCount);

    clock_t start = clock();
    for (int i = 0; i < countOfCrossVal; i++) {
        
        cout << "countOfCrossVal = " << countOfCrossVal << endl;

        X_test.resize(0, 0);
        X_train2.resize(0, 0);
        X_train3.resize(0, 0);


        X_test = X.block(crossValCount * i, 0, crossValCount, X.cols());

        if (i != 0) {
            X_train2 = X.block(0, 0, crossValCount*i, X.cols());
            X_train3 = X.block(crossValCount * (i + 1), 0, X.rows() - crossValCount * (i + 1), X.cols());

            Eigen::SparseMatrix<float, ColMajor> M(X_train2.rows() + X_train3.rows(), X_train3.cols());
            M.reserve(X_train2.nonZeros() + X_train3.nonZeros());
            for (Index c = 0; c < X_train2.cols(); ++c) {
                M.startVec(c);
                for (Eigen::SparseMatrix<float, ColMajor>::InnerIterator itL(X_train2, c); itL; ++itL)
                    M.insertBack(itL.row(), c) = itL.value();
                for (Eigen::SparseMatrix<float, ColMajor>::InnerIterator itC(X_train3, c); itC; ++itC)
                    M.insertBack(itC.row() + X_train2.rows(), c) = itC.value();
            }
            M.finalize();

            X_train = M;
        } else {
            X_train = X.block(crossValCount, 0, X.rows() - crossValCount, X.cols());
        }

        long int indexTest = 0;
        long int indexTran = 0;

        for (int j = 0; j < Y.rows(); j++) {
            if (j < crossValCount * i || j >= crossValCount * (i + 1)) {
                Y_train(indexTran) = Y(j);
                ++indexTran;
            } else {
                Y_test(indexTest) = Y(j);
                ++indexTest;
            }
        }

        cout << "Y_train.rows() = " << Y_train.rows() << endl;
        cout << "Y_test.rows() = " << Y_test.rows() << endl;

        cout << "X_train.rows() = " << X_train.rows() << endl;
        cout << "X_test.rows() = " << X_test.rows() << endl;
        cout << "bach_size = " << bach_size << endl;
        FM model(learning_rate, numEpoh, bach_size, _k, 2649420, 17770);
        clock_t start2 = clock();
        model.fit(X_train, Y_train);
        clock_t end2 = clock();

        double seconds2 = (double) (end2 - start2) / CLOCKS_PER_SEC;
        cout << " time to fit = " << seconds2 << " seconds" << endl << endl;


        VectorXf Y_pred = model.predict(X_train);
        double result_RMSE = RMSE_metric::calculateMetric(Y_pred, Y_train);

        cout << "result RMSE trening iteretion #" << i << " = " << result_RMSE << endl;

        auto Y_pred_test = model.predict(X_test);
        float result_RMSE_test = RMSE_metric::calculateMetric(Y_pred_test, Y_test);

        cout << "result RMSE test iteretion #" << i << " = " << result_RMSE_test << endl;

        RMSE_results.push_back(result_RMSE_test);
//        cout<<"!!!!!!!!!!!!!!!!!!"<<endl;
        all_W.push_back(model.getW());
       
    }
    clock_t end = clock();

    double seconds = (double) (end - start) / CLOCKS_PER_SEC;
    cout << " time = " << seconds << " seconds" << endl << endl;

    double RMSE_M = 0;

    double RMSE_sig = 0;

    Statistic::findeStatistic(RMSE_results, RMSE_M, RMSE_sig);

    std::ofstream outFile;

    cout << "all RMSE Mean = " << RMSE_M << endl;
    cout << "all RMSE Sigma = " << RMSE_sig << endl;


    std::ofstream myfile;
    myfile.open("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/NetflixPrize_Home_FM//Result_table2.csv");
    myfile << ",1,2,3,4,5,E,SD,\n";
    myfile << "RMSE," << (RMSE_results[0]) << "," << (RMSE_results[1]) << "," << (RMSE_results[2]) << "," << (RMSE_results[3]) << "," << (RMSE_results[4]) << "," << RMSE_M << "," << RMSE_sig << ",\n";

        for (int i = 0; i < all_W[0].size(); i++) {
            float W_i_M = 0;
            float W_i_Sig = 0;
            for (int k = 0; k < all_W.size(); ++k) {
                W_i_M += all_W[k][i];
                W_i_Sig += all_W[k][i] * all_W[k][i];
            }
            W_i_M = W_i_M / all_W.size();
            W_i_Sig = sqrt(W_i_Sig / all_W.size() - W_i_M * W_i_M);
            myfile << "X["<<i<<"}" << "," << all_W[0][i] << "," << all_W[1][i] << "," << all_W[2][i] << "," << all_W[3][i] << "," << all_W[4][i] << "," << W_i_M << "," << W_i_Sig << ",\n";
        }

    myfile.close();
}

float CrossValScore::getMeanRMSE() {
    return this->countOfCrossVal;
}

