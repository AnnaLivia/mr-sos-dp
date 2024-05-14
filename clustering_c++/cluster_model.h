#ifndef CLUSTERING_CLUSTER_MODEL_H
#define CLUSTERING_CLUSTER_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class GRBMatrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    GRBMatrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};


class cluster_model {

	protected:
	int status;
	int n, p, k;

	std::string get_x_variable_name(int i, int h);
	std::string get_y_variable_name(int i, int j, int h);
	std::string get_point_constraint_name(int i);
	std::string get_part_constraint_name(int c, int h);
	std::string get_edge_constraint_name(int i, int j, int h);
	
	public:
	virtual int get_n_constraints() = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_edge_constraints() = 0;
	virtual void add_min_constraints(arma::mat dist) = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual arma::mat get_x_solution() = 0;
};

class cluster_gurobi_model : public cluster_model {

private:
	GRBEnv *env;
	GRBModel model;
	GRBMatrix<GRBVar> X;
	GRBMatrix<GRBVar> Y;
	GRBMatrix<GRBVar> Z;

	GRBMatrix<GRBVar> create_X_variables(GRBModel &model);
	GRBMatrix<GRBVar> create_Y_variables(GRBModel &model, arma::mat dist);
	GRBMatrix<GRBVar> create_Z_variables(GRBModel &model);


public:
	cluster_gurobi_model(GRBEnv *env, int n, int p, int k, arma::mat dist);
	virtual int get_n_constraints();
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_edge_constraints();
	virtual void add_min_constraints(arma::mat dist);
	virtual void optimize();
	virtual double get_value();
	virtual arma::mat get_x_solution();
};

#endif //CLUSTERING_CLUSTER_MODEL_H
