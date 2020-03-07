#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "Stats.h"
using namespace std;
using namespace Eigen;

class GaussianNB
{
private:
	int K;
	RowVectorXd* MUs;
	MatrixXd* sigmas;
public:
	GaussianNB();
	~GaussianNB();
	void fit(const MatrixXd& X, const VectorXd& Y);
	VectorXd predict(const MatrixXd& X);
};

namespace gnb
{
	int n_classes(const VectorXd& Y);

	vector<MatrixXd> split_by_class(const MatrixXd& X, const VectorXd& Y);

	vector<vector<int>> to_groups(const VectorXd& Y);
}

GaussianNB::GaussianNB() : K(0), MUs(nullptr), sigmas(nullptr) {}

GaussianNB::~GaussianNB()
{
	delete[] MUs;
	delete[] sigmas;
}

void GaussianNB::fit(const MatrixXd& X, const VectorXd& Y)
{
	K = gnb::n_classes(Y);
	MUs = new RowVectorXd[K];
	sigmas = new MatrixXd[K];

	vector<MatrixXd> xs = gnb::split_by_class(X, Y);
	for (int i = 0; i < K; i++)
	{
		MUs[i] = stats::mean(xs[i], 1);
		sigmas[i] = stats::cov(xs[i].transpose());
	}
}

int gnb::n_classes(const VectorXd& Y)
// assuming a class vector has consecutive integers starting from zero
{
	return *std::max_element(Y.data(), Y.data() + Y.size()) + 1;
}

vector<MatrixXd> gnb::split_by_class(const MatrixXd& X, const VectorXd& Y)
{
	vector<vector<int>> groups = to_groups(Y);

	vector<MatrixXd> splits;
	for (const auto& group : groups)
	{
		MatrixXd split(group.size(), X.cols());
		for (int i = 0; i < group.size(); i++)
			split.row(i) = X.row(group[i]);
		splits.push_back(split);
	}
	return splits;
}

vector<vector<int>> gnb::to_groups(const VectorXd& Y)
{
	vector<vector<int>> groups(n_classes(Y), vector<int>());
	for (int i = 0; i < Y.size(); i++)
		groups[Y[i]].push_back(i);
	return groups;
}

VectorXd GaussianNB::predict(const MatrixXd& X)
{
	VectorXd labels(X.rows());
	for (int i = 0; i < X.rows(); i++)
	{
		vector<double> temp(K);
		for (int j = 0; j < K; j++)
			temp[j] = stats::multivariate_normal(X.row(i), MUs[j], sigmas[j]);
		labels[i] = std::distance(temp.begin(), std::max_element(temp.begin(), temp.end()));
	}
	return labels;
}