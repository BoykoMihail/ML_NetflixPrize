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


void read_training_text(char* data, Eigen::SparseMatrix<float, RowMajor> &X, VectorXf &Y);

int main(int argc, char** argv) {


    Eigen::SparseMatrix<float, RowMajor> X;
    VectorXf Y;

    read_training_text("/home/boyko_mihail/NetBeansProjects/course_Ml/Boyko/NetflixPrize_Home_FM/data/dataReiting/training_set", X, Y);

    CrossValScore crossValModel(2.6, 37, 2000000, 5, 5, 2649420, 17770);

    crossValModel.fit(X, Y);

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

void read_training_text(char* data, Eigen::SparseMatrix<float, RowMajor> &X, VectorXf &Y) {
    unsigned int maxUsers = 2649420;
    unsigned int maxFilms = 17770;
    unsigned int index = 0;
    typedef Eigen::Triplet<float> T;
    std::vector<float> yVector(0);
    std::vector<T> tripletList;
    for (unsigned int i = 1; i < maxFilms; ++i) {

        std::string buf(data);
        string fileName = getFileName(i);
        cout << fileName << endl;
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
                    int raiting = stod(tokens[1]);
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

    Y = VectorXf::Map(yVector.data(), yVector.size());

    X.resize(0, 0);

    X.resize(index, maxFilms + maxUsers);
    X.setFromTriplets(tripletList.begin(), tripletList.end());
    tripletList.clear();
    std::cout << X.rows() << endl;
    std::cout << X.nonZeros() << endl;


}



