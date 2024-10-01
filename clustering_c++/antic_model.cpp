#include "antic_model.h"

template<class T>
AMatrix<T>::AMatrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &AMatrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T AMatrix<T>::operator()(size_t row, size_t col) const {
	return data[row*cols+col];
}


std::string antic_model::get_x_variable_name(int i, int h){
	std::ostringstream os;
	os << "x" << i << "_" << h;
	return os.str();
}

std::string antic_model::get_y_variable_name(int l){
	std::ostringstream os;
	os << "y" << l;
	return os.str();
}


antic_gurobi_model::antic_gurobi_model(GRBEnv *env, int n, int p, int d) : model(*env), X(n,p), Y(1,d)

 {
	this->n = n;
	this->p = p;
	this->d = d;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->Y = create_Y_variables(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
    this->model.set("TimeLimit", "300");
    //this->model.set("Presolve", 1);
	model.set(GRB_StringParam_LogFile, result_path + "_GRB_anti_log.txt");
}


AMatrix<GRBVar> antic_gurobi_model::create_X_variables(GRBModel &model) {
    AMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < p; h++) {
			std::string name = get_x_variable_name(i, h);
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
        }
    }
    return X;
}

AMatrix<GRBVar> antic_gurobi_model::create_Y_variables(GRBModel &model) {
    AMatrix<GRBVar> Y(1,d);
    for (int l = 0; l < d; l++) {
        std::string name = get_y_variable_name(l);
        Y(0, l) = model.addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, name);
    }
    return Y;
}

void antic_gurobi_model::add_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        model.addConstr(lhs_sum == 1);
    }
}

void antic_gurobi_model::add_part_constraints() {
    for (int h = 0; h < p; h++) {
        GRBLinExpr lhs_sum = 0;
        for (int i = 0; i < n; i++)
            lhs_sum += X(i, h);
    	int points = floor(n/p);
    	if (h < n % p)
    		points += 1;
    	model.addConstr(lhs_sum == points);
    }
}

void antic_gurobi_model::add_dev_constraints(std::vector<std::vector<double>> &data) {
    for (int h = 0; h < p; h++) {
    	int points = floor(n/p);
    	if (h < n % p)
    		points += 1;
    	for (int l = 0; l < d; l++) {
        	GRBLinExpr lhs_sum = 0;
    		double centroid = 0;
        	for (int i = 0; i < n; i++) {
            	lhs_sum += data[i][l]*X(i, h);
        		centroid += data[i][l];
            }
            model.addConstr(Y(0,l) >= lhs_sum/points - centroid/n);
            model.addConstr(Y(0,l) >= - lhs_sum/points + centroid/n);
		}
    }
}

void antic_gurobi_model::optimize(){
	try {
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
		model.write(result_path + "_GRB_anti.lp");
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}

int antic_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

void antic_gurobi_model::get_x_solution(std::vector<std::vector<int>> &sol) {
	for (int h = 0; h < p; h++) {
		int points = floor(n/p);
		if (h < n % p)
			points += 1;
		sol[h].resize(points);
		int nc = 0;
		for (int i = 0; i < n; i++) {
        	if (X(i, h).get(GRB_DoubleAttr_X) > 0.8) {
        		sol[h][nc] = i;
        		nc++;
        	}
		}
	}
}

double antic_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}

double antic_gurobi_model::get_gap(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_MIPGap);
}