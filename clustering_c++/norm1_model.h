#ifndef CLUSTERING_NORM_MODEL_H
#define CLUSTERING_NORM_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class NMatrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    NMatrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};

class norm_model {

	protected:
	int status;
	int n, p, d;
	std::vector<int> cls_points;

	std::string get_x_variable_name(int i, int h);
	std::string get_y_variable_name(int l);
	
	public:
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_dev_constraints(std::vector<std::vector<double>> &data, std::vector<double> &center) = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual double get_gap() = 0;
	virtual std::vector<std::vector<int>> get_x_solution() = 0;
};

class norm_gurobi_model : public norm_model {

private:
	GRBEnv *env;
	GRBModel model;
	NMatrix<GRBVar> X;
	NMatrix<GRBVar> Y;
	GRBVar Z;

	NMatrix<GRBVar> create_X_variables(GRBModel &model);
	NMatrix<GRBVar> create_Y_variables(GRBModel &model);
	GRBVar create_Z_variable(GRBModel &model);


public:
	norm_gurobi_model(GRBEnv *env, int n, int p, int d, std::vector<int> cls_points);
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_dev_constraints(std::vector<std::vector<double>> &data, std::vector<double> &center);
	virtual void optimize();
	virtual double get_value();
	virtual double get_gap();
	virtual std::vector<std::vector<int>> get_x_solution();
};

#endif //CLUSTERING_NORM_MODEL_H
