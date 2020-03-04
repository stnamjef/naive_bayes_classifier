#pragma once
#include <iostream>
#include <vector>
#include <cmath>
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
	void to_groups(const VectorXd& vec, vector<vector<int>>& groups);

	void split_by_class(const MatrixXd& X, vector<vector<int>>& groups, MatrixXd*& xs);
}

GaussianNB::GaussianNB() : K(0), MUs(nullptr), sigmas(nullptr) {}

GaussianNB::~GaussianNB()
{
	delete[] MUs;
	delete[] sigmas;
}

void GaussianNB::fit(const MatrixXd& X, const VectorXd& Y)
{
	vector<vector<int>> groups;
	gnb::to_groups(Y, groups);

	K = (int)groups.size();

	MUs = new RowVectorXd[K];
	sigmas = new MatrixXd[K];

	MatrixXd* xs = new MatrixXd[K];
	gnb::split_by_class(X, groups, xs);

	for (int i = 0; i < K; i++)
	{
		MUs[i] = stats::mean(xs[i], 1);
		sigmas[i] = stats::cov(xs[i].transpose());
	}

	delete[] xs;
}

void gnb::to_groups(const VectorXd& vec, vector<vector<int>>& groups)
{
	for (int i = 0; i < vec.size(); i++)
	{
		if (vec[i] >= groups.size())
			groups.push_back(vector<int>(1, i));
		else
			groups[vec[i]].push_back(i);
	}
}

void gnb::split_by_class(const MatrixXd& X, vector<vector<int>>& groups, MatrixXd*& xs)
{
	for (int i = 0; i < groups.size(); i++)
	{
		MatrixXd temp(groups[i].size(), X.cols());
		for (int j = 0; j < groups[i].size(); j++)
			temp.row(j) = X.row(groups[i][j]);
		xs[i] = temp;
	}
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