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
	std::vector<std::vector<double>> dist;
	std::vector<std::vector<int>> cls_points;

public:
	virtual int get_n_constraints() = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_bound_constraints() = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual double get_gap() = 0;
	virtual void get_x_solution(std::vector<std::vector<int>> &sol) = 0;
};

class glover_gurobi_model : public glover_model {

private:
	GRBEnv *env;
	GRBModel model;
	GWMatrix<GRBVar> X;
	GWMatrix<GRBVar> W;
	GRBVar Z;

	GWMatrix<GRBVar> create_X_variables(GRBModel &model);
	GWMatrix<GRBVar> create_W_variables(GRBModel &model);
	GRBVar create_Z_variable(GRBModel &model);


public:
	glover_gurobi_model(GRBEnv *env, int n, int p, int k, std::vector<std::vector<double>> dist, std::vector<std::vector<int>> cls_points);
	virtual int get_n_constraints();
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_bound_constraints();
	virtual void optimize();
	virtual double get_value();
	virtual double get_gap();
	virtual void get_x_solution(std::vector<std::vector<int>> &sol);
};

class cls_model {

	protected:
	int status;
	int n, p, k;
	std::vector<std::vector<double>> dist;
	std::vector<int> cls;

public:
	virtual void add_cls_point_constraints() = 0;
	virtual void add_cls_part_constraints() = 0;
	virtual void add_cls_bound_constraints() = 0;
	virtual void optimize_cls() = 0;
	virtual double get_cls_value() = 0;
	virtual double get_cls_gap() = 0;
	virtual void get_cls_solution(std::vector<std::vector<int>> &sol) = 0;
};


class glover_cls_model : public cls_model {

private:
	GRBEnv *env;
	GRBModel model;
	GWMatrix<GRBVar> X;
	GWMatrix<GRBVar> W;

	GWMatrix<GRBVar> create_Xcls_variables(GRBModel &model);
	GWMatrix<GRBVar> create_Wcls_variables(GRBModel &model);


public:
	glover_cls_model(GRBEnv *env, int n, int p, int k, std::vector<std::vector<double>> dist, std::vector<int> cls);
	virtual void add_cls_point_constraints();
	virtual void add_cls_part_constraints();
	virtual void add_cls_bound_constraints();
	virtual void optimize_cls();
	virtual double get_cls_value();
	virtual double get_cls_gap();
	virtual void get_cls_solution(std::vector<std::vector<int>> &sol);
};

#endif //CLUSTERING_GLOVER_MODEL_H
