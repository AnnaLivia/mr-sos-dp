#ifndef CLUSTERING_COMB_MODEL_H
#define CLUSTERING_COMB_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class CMatrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    CMatrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};

class comb_model {

	protected:
	int status;
	int n, p, d;
	std::vector<int> cls_points;

	std::string get_x_variable_name(int i, int h);
	std::string get_y_variable_name(int l);
	std::string get_point_constraint_name(int i);
	std::string get_part_constraint_name(int h);
	std::string get_edge_constraint_name(int i, int j, int h);
	
	public:
	virtual int get_n_constraints() = 0;
	virtual void set_objective_function(std::vector<int> &cls_points, std::vector<std::vector<double>> &dist) = 0;
	virtual void add_dev_constraints(std::vector<std::vector<double>> &data, std::vector<double> &center) = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_edge_constraints() = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual double get_gap() = 0;
	virtual std::vector<std::vector<int>> get_x_solution() = 0;
};

class comb_gurobi_model : public comb_model {

private:
	GRBEnv *env;
	GRBModel model;
	CMatrix<GRBVar> X;
	CMatrix<GRBVar> Y;
	GRBVar Z;

	CMatrix<GRBVar> create_X_variables(GRBModel &model);
	CMatrix<GRBVar> create_Y_variables(GRBModel &model);
	GRBVar create_Z_variable(GRBModel &model);


public:
	comb_gurobi_model(GRBEnv *env, int n, int p, int d, std::vector<int> cls_points);
	virtual int get_n_constraints();
	virtual void set_objective_function(std::vector<int> &cls_points, std::vector<std::vector<double>> &dist);
	virtual void add_dev_constraints(std::vector<std::vector<double>> &data, std::vector<double> &center);
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_edge_constraints();
	virtual void optimize();
	virtual double get_value();
	virtual double get_gap();
	virtual std::vector<std::vector<int>> get_x_solution();
};

#endif //CLUSTERING_COMB_MODEL_H
