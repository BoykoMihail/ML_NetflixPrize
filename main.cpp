/* 
 * File:   main.cpp
 * Author: boyko_mihail
 *
 * Created on 13 октября 2019 г., 17:44
 */

#include <cstdlib>

#include "FM.h"
#include "CrossValScore.h"

using namespace std;

/*
 * 
 */



void read_training_text(char* data, Eigen::SparseMatrix<double> &X, VectorXd &Y);

int main(int argc, char** argv) {

//    omp_set_num_threads(4);
    Eigen::setNbThreads(4);
    cout<<"Eigen::nbThreads() = "<<Eigen::nbThreads()<<endl;;
    Eigen::SparseMatrix<double> X;
    VectorXd Y;
    read_training_text("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/NetflixPrize_Home_FM/data/dataReiting/training_set", X, Y);



    FM model(0.02, 10, 100000, 3, 5, 2649420, 17770);
    CrossValScore crossValModel(model, 5, 2649420, 17770);

    crossValModel.fit(X, Y);
    // f->fit("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/NetflixPrize_Home_FM/data/dataReiting/training_set");
    return 0;
}

vector<string>& split(const string &s, char delim, vector<string> &elems) {
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim) {
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

string getFileName(unsigned int item) {
    string start = "/mv_";
    unsigned int n = 1000000;
    while (item / n == 0) {
        start.append("0");
        n /= 10;
    }
    while (n != 0) {
        start.append(std::to_string(item / n));

        item %= n;
        n /= 10;
    }
    start.append(".txt");
    return start;
}

void read_training_text(char* data, Eigen::SparseMatrix<double> &X, VectorXd &Y) {
    unsigned int maxUsers = 2649420;
    unsigned int maxFilms = 17770;
    unsigned int index = 0;
    typedef Eigen::Triplet<double> T;
    std::vector<double> yVector(0);
    std::vector<T> tripletList;
//    Eigen::SparseMatrix<double> NewX(110000000, maxFilms + maxUsers);
//    NewX.reserve(220000000);
    for (unsigned int i = 1; i < maxFilms/2; ++i) {

        std::string buf(data);
        string fileName = getFileName(i);
      //  cout << fileName << endl;
        buf.append(fileName);
        ifstream file(buf);
        if (file.is_open()) {
            string line;
            getline(file, line);
            std::vector<string> filmsIds = split(line, ':');
            unsigned int filmsId = stod(filmsIds[0]);
            while (getline(file, line)) {
                vector<string> tokens = split(line, ',');

                if (tokens.size() == 3) {
                    unsigned int user = stod(tokens[0]);
                    unsigned int raiting = stod(tokens[1]);
                    //                    NewX.insert(index, user) = 1;
                    //                    NewX.insert(index, filmsId + maxUsers) = 1;
                    tripletList.push_back(T(index, user, 1));
                    tripletList.push_back(T(index, filmsId + maxUsers, 1));
                    yVector.push_back(raiting);
                    ++index;

                }
            }
            file.close();
        }
    }


    cout << "index = " << index << endl;
    cout << "maxFilms + maxUsers = " << maxFilms + maxUsers << endl;
    Y = VectorXd::Map(yVector.data(), yVector.size());
    X.resize(0, 0);
    X.resize(index , maxFilms + maxUsers );
    X.setFromTriplets(tripletList.begin(), tripletList.end());
    //    X = NewX;
//
//    std::cout <<"max_item = "<< max_item << endl;
//    std::cout << "max_user = "<< max_user << endl;
    
    std::cout << X.rows() << endl;
    std::cout << X.nonZeros() << endl;

}



