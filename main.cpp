#include <iostream>
#include "manage_file.h"
#include "naive_bayes.h"
#include "model_selection.h"
using namespace std;
using namespace Eigen;

int main()
{
	MatrixXd df;
	VectorXd labels;

	read_csv("data/iris.csv", df, labels, 150, 5);

	GaussianNB model;
	double accuracy = evaluate_model(model, df, labels, 5);
	cout << "Model accuracy: " << accuracy * 100 << "%" << endl;

	return 0;
}