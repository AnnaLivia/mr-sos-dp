#ifndef CLUSTERING_MOUNT_MODEL_H
#define CLUSTERING_MOUNT_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class MMatrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    MMatrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};


class mount_model {

	protected:
	int status;
	int n, p, k, m;
	arma::mat dist;
    std::unordered_map<int, std::unordered_map<int, arma::mat>> sol_cls;

	std::string get_x_variable_name(int c1, int h1, int t);
	std::string get_y_variable_name(int c1, int h1, int c2, int h2, int t);

	public:
	virtual int get_n_constraints() = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_cls_constraints() = 0;
	virtual void add_edge_constraints() = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual std::map<int, arma::vec> get_x_solution() = 0;
};

class mount_gurobi_model : public mount_model {

private:
	GRBEnv *env;
	GRBModel model;
	MMatrix<GRBVar> X;
	MMatrix<GRBVar> Y;

	MMatrix<GRBVar> create_X_variables(GRBModel &model);
	MMatrix<GRBVar> create_Y_variables(GRBModel &model);


public:
	mount_gurobi_model(GRBEnv *env, int n, int p, int k, int m, arma::mat &dist, std::unordered_map<int, std::unordered_map<int, arma::mat>> &sol_cls);
	virtual int get_n_constraints();
	virtual void add_point_constraints();
	virtual void add_cls_constraints();
	virtual void add_edge_constraints();
	virtual void optimize();
	virtual double get_value();
	virtual std::map<int, arma::vec> get_x_solution();
};

#endif //CLUSTERING_MOUNT_MODEL_H
