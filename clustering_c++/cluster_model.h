#ifndef CLUSTERING_CLUSTER_MODEL_H
#define CLUSTERING_CLUSTER_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class CLMatrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    CLMatrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};


class cluster_model {

	protected:
	int status;
	int n, p, k,size_y;
	std::vector<std::vector<double>> dist;
	std::vector<std::vector<int>> cls_points;
	
	public:
	virtual int get_n_constraints() = 0;
	virtual void add_point_constraints() = 0;
	virtual void add_part_constraints() = 0;
	virtual void add_edge_constraints() = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual double get_gap() = 0;
	virtual void get_x_solution(std::vector<std::vector<int>> &sol) = 0;
};

class cluster_gurobi_model : public cluster_model {

private:
	GRBEnv *env;
	GRBModel model;
	CLMatrix<GRBVar> X;
	CLMatrix<GRBVar> Y;
	GRBVar Z;

	CLMatrix<GRBVar> create_X_variables(GRBModel &model);
	CLMatrix<GRBVar> create_Y_variables(GRBModel &model);
	GRBVar create_Z_variable(GRBModel &model);


public:
	cluster_gurobi_model(GRBEnv *env, int n, int p, int k, int size_y, std::vector<std::vector<double>> dist, std::vector<std::vector<int>> cls_points);
	virtual int get_n_constraints();
	virtual void add_point_constraints();
	virtual void add_part_constraints();
	virtual void add_edge_constraints();
	virtual void optimize();
	virtual double get_value();
	virtual double get_gap();
	virtual void get_x_solution(std::vector<std::vector<int>> &sol);
};


class edge_cls_model {

	protected:
	int status;
	int n, p, k,size_y;
	std::vector<std::vector<double>> dist;
	std::vector<int> cls;
	arma::mat cls_data;

	public:
	virtual void add_cls_point_constraints() = 0;
	virtual void add_cls_part_constraints() = 0;
	virtual void add_cls_edge_constraints() = 0;
	virtual void optimize_cls() = 0;
	virtual double get_cls_value() = 0;
	virtual double get_cls_gap() = 0;
	virtual void get_cls_solution(std::vector<std::vector<int>> &sol) = 0;
};

class gurobi_model : public edge_cls_model {

private:
	GRBEnv *env;
	GRBModel model;
	CLMatrix<GRBVar> X;
	CLMatrix<GRBVar> Y;

	CLMatrix<GRBVar> create_Xcls_variables(GRBModel &model);
	CLMatrix<GRBVar> create_Ycls_variables(GRBModel &model);

public:
	gurobi_model(GRBEnv *env, int n, int p, int k, std::vector<std::vector<double>> dist, std::vector<int> cls, arma::mat cls_data);
	virtual void add_cls_point_constraints();
	virtual void add_cls_part_constraints();
	virtual void add_cls_edge_constraints();
	virtual void optimize_cls();
	virtual double get_cls_value();
	virtual double get_cls_gap();
	virtual void get_cls_solution(std::vector<std::vector<int>> &sol);
};

// Callback class to add violated inequalities at the root node
class MyCallback : public GRBCallback {

private:
	int n; int p;
	CLMatrix<GRBVar> X;
	CLMatrix<GRBVar> Y;
public:
    MyCallback(int n, int p, CLMatrix<GRBVar> X, CLMatrix<GRBVar> Y);

protected:
    void callback();
};

#endif //CLUSTERING_CLUSTER_MODEL_H
