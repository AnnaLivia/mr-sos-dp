#ifndef CLUSTERING_GLOVER_MODEL_H
#define CLUSTERING_GLOVER_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class GWMatrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    GWMatrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};


class glover_model {

	protected:
	int status;
	int n, p, k;
	arma::mat dist;
	arma::mat lb;
	arma::mat ub;

	std::string get_x_variable_name(int i, int h);
	std::string get_w_variable_name(int i, int h);
	std::string get_point_constraint_name(int i);
	std::string get_part_constraint_name(int c, int h);
	std::string get_lb_constraint_name(int i, int h);
	std::string get_ub_constraint_name(int i, int h);

	public:
	virtual int get_n_constraints() = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_bound_constraints() = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual arma::mat get_x_solution() = 0;
};

class glover_gurobi_model : public glover_model {

private:
	GRBEnv *env;
	GRBModel model;
	GWMatrix<GRBVar> X;
	GWMatrix<GRBVar> W;

	GWMatrix<GRBVar> create_X_variables(GRBModel &model);
	GWMatrix<GRBVar> create_W_variables(GRBModel &model);


public:
	glover_gurobi_model(GRBEnv *env, int n, int p, int k, arma::mat dist, arma::mat lb, arma::mat ub);
	virtual int get_n_constraints();
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_bound_constraints();
	virtual void optimize();
	virtual double get_value();
	virtual arma::mat get_x_solution();
};

#endif //CLUSTERING_GLOVER_MODEL_H
