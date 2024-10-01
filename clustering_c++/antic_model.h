#ifndef CLUSTERING_ANTIC_MODEL_H
#define CLUSTERING_ANTIC_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class AMatrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    AMatrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};

class antic_model {

	protected:
	int status;
	int n, p, d;
	std::vector<int> cls_points;

	std::string get_x_variable_name(int i, int h);
	std::string get_y_variable_name(int l);

	public:
	virtual int get_n_constraints() = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_dev_constraints(std::vector<std::vector<double>> &data) = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual double get_gap() = 0;
	virtual void get_x_solution(std::vector<std::vector<int>> &sol) = 0;
};

class antic_gurobi_model : public antic_model {

private:
	GRBEnv *env;
	GRBModel model;
	AMatrix<GRBVar> X;
	AMatrix<GRBVar> Y;
	GRBVar Z;

	AMatrix<GRBVar> create_X_variables(GRBModel &model);
	AMatrix<GRBVar> create_Y_variables(GRBModel &model);


public:
	antic_gurobi_model(GRBEnv *env, int n, int p, int d);
	virtual int get_n_constraints();
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_dev_constraints(std::vector<std::vector<double>> &data);
	virtual void optimize();
	virtual double get_value();
	virtual double get_gap();
	virtual void get_x_solution(std::vector<std::vector<int>> &sol);
};

#endif //CLUSTERING_ANTIC_MODEL_H
