#include "norm1_model.h"

template<class T>
NMatrix<T>::NMatrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &NMatrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T NMatrix<T>::operator()(size_t row, size_t col) const {
	return data[row*cols+col];
}


norm_gurobi_model::norm_gurobi_model(GRBEnv *env, int n, int p, int d, std::vector<int> cls_points) : model(*env), X(n,p), Y(p,d), Z()

 {
	this->n = n;
	this->p = p;
	this->d = d;
	this->cls_points = cls_points;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->Y = create_Y_variables(this->model);
	this->Z = create_Z_variable(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
    this->model.set("TimeLimit", "300");
	//this->model.set("Presolve", 1);
	model.set(GRB_StringParam_LogFile, result_path + "_GRB_norm1_log.txt");
}


NMatrix<GRBVar> norm_gurobi_model::create_X_variables(GRBModel &model) {
    NMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++)
        for (int h = 0; h < p; h++)
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, "x_" + std::to_string(i) + "_" + std::to_string(h));
    return X;
}

NMatrix<GRBVar> norm_gurobi_model::create_Y_variables(GRBModel &model) {
    NMatrix<GRBVar> Y(p,d);
    for (int h = 0; h < p; h++)
    	for (int l = 0; l < d; l++)
       		Y(h, l) = model.addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, "y_" + std::to_string(h) + "^" + std::to_string(l));
    return Y;
}

GRBVar norm_gurobi_model::create_Z_variable(GRBModel &model) {
	GRBVar Z = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Z");
	return Z;
}

void norm_gurobi_model::add_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        model.addConstr(lhs_sum == 1);
    }
}

void norm_gurobi_model::add_part_constraints() {
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

void norm_gurobi_model::add_dev_constraints(std::vector<std::vector<double>> &data, std::vector<double> &center) {
    for (int h = 0; h < p; h++) {
    	int points = floor(n/p);
    	if (h < n % p)
    		points += 1;
    	for (int l = 0; l < d; l++) {
    		GRBLinExpr lhs_sum = 0;
    		std::vector<double> diff(n);
    		for (int i = 0; i < n; i++) {
    			lhs_sum += data[cls_points[i]][l]*X(i, h);
    			diff.push_back(data[cls_points[i]][l] - center[l]*points);
    		}
    		std::sort(diff.begin(), diff.end(), [](double a, double b) {
				return std::abs(a) < std::abs(b);  // Comparator to sort by absolute values
			});
    		double abs_sum = 0;
    		for (int i = 0; i < points; ++i)
    			abs_sum += diff[i];
            model.addConstr(Y(h,l) >= lhs_sum - center[l]*points);
            model.addConstr(Y(h,l) >= - lhs_sum + center[l]*points);
            model.addConstr(Y(h,l) >= abs_sum);
            //model.addConstr(Z >= Y(h,l));
		}
    }
}

void norm_gurobi_model::optimize(){
	try {
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
		model.write(result_path + "_GRB_norm.lp");
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}

std::vector<std::vector<int>> norm_gurobi_model::get_x_solution() {
	std::vector<std::vector<int>> sol_cls(p);
	for (int h = 0; h < p; h++) {
		int points = floor(n/p);
		if (h < n % p)
			points += 1;
		sol_cls[h].resize(points);
		int nc = 0;
		for (int i = 0; i < n; i++) {
        	if (X(i, h).get(GRB_DoubleAttr_X) > 0.8) {
        		sol_cls[h][nc] = cls_points[i];
        		nc++;
        	}
		}
	}
	return sol_cls;
}

double norm_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}

double norm_gurobi_model::get_gap(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_MIPGap);
}

