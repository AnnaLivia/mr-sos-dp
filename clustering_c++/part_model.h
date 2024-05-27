#ifndef CLUSTERING_PART_MODEL_H
#define CLUSTERING_PART_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class GRB_Matrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    GRB_Matrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};


class part_model {

	protected:
	int status;
	int n, p, c;
	arma::mat dist;
	
	public:
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_edge_constraints() = 0;
	virtual void add_min_constraints() = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual arma::mat get_x_solution() = 0;
};

class part_gurobi_model : public part_model {

private:
	GRBEnv *env;
	GRBModel model;
	GRB_Matrix<GRBVar> X;
	GRB_Matrix<GRBVar> Y;
	GRB_Matrix<GRBVar> Z;

	GRB_Matrix<GRBVar> create_X_variables(GRBModel &model);
	GRB_Matrix<GRBVar> create_Y_variables(GRBModel &model);
	GRB_Matrix<GRBVar> create_Z_variables(GRBModel &model);


public:
	part_gurobi_model(GRBEnv *env, int n, int p, int c, arma::mat dist);
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_edge_constraints();
	virtual void add_min_constraints();
	virtual void optimize();
	virtual double get_value();
	virtual arma::mat get_x_solution();
};

#endif //CLUSTERING_PART_MODEL_H
