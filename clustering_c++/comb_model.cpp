#include "comb_model.h"

template<class T>
Matrix<T>::Matrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &Matrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T Matrix<T>::operator()(size_t row, size_t col) const {
    return data[row*cols+col];
}



std::string comb_model::get_x_variable_name(int i, int h){
	std::ostringstream os;
	os << "x" << i << "_" << h;
	return os.str();
}

std::string comb_model::get_y_variable_name(int i, int j, int h){
	std::ostringstream os;
	os << "y" << i << "_" << j << "_p" << h;
	return os.str();
}

std::string comb_model::get_point_constraint_name(int i){
	std::ostringstream os;
	os << "point" << i;
	return os.str();
}

std::string comb_model::get_part_constraint_name(int h){
	std::ostringstream os;
	os << "part" << h;
	return os.str();
}

/*
std::string comb_model::get_edge_constraint_name(int i, int j, int h){
	std::ostringstream os;
	os << "y" << i << " " << j << "_p" << h;
	return os.str();
}
*/


comb_gurobi_model::comb_gurobi_model(GRBEnv *env, int n, int p) : model(*env), X(n,p) {
	this->n = n;
	this->p = p;
	this->m = n*(n-1)/2;
	this->env = env;
	this->X = create_X_variables(this->model);
	//this->Y = create_Y_variables(this->model);
    this->model.set("OutputFlag", "1");
//    this->model.set("Threads", "4");
    this->model.set("TimeLimit", "60");
    //this->model.set("Presolve", std::to_string(lp_solver_presolve));
}

Matrix<GRBVar> comb_gurobi_model::create_X_variables(GRBModel &model) {
    Matrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < p; h++) {
			std::string name = get_x_variable_name(i, h);
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
        }
    }
    return X;
}

/*
Matrix<GRBVar> comb_gurobi_model::create_Y_variables(GRBModel &model) {
    Matrix<GRBVar> Y(m, p);
    for (int h = 0; h < m; h++) {
        int s = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                std::string name = get_y_variable_name(i, j, h);
                Y(s, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
                s++;
            }
        }
    }
    return Y;
}
*/

void comb_gurobi_model::add_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        std::string name = get_point_constraint_name(i);
        model.addConstr(lhs_sum == 1, name);
    }
}

void comb_gurobi_model::add_part_constraints() {
    for (int h = 0; h < p; h++) {
        GRBLinExpr lhs_sum = 0;
        for (int i = 0; i < n; i++)
            lhs_sum += X(i, h);
        std::string name = get_part_constraint_name(h);
        //model.addConstr(lhs_sum <= n/(p - 1), name);
        //model.addConstr(lhs_sum >= n/(p + 1), name);
        model.addConstr(lhs_sum <= n/p + 1, name);
    }
}

/*
void comb_gurobi_model::add_edge_constraints() {
    int s;
    for (int h = 0; h < p; h++) {
        s = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
                std::string name = get_edge_constraint_name(i, j, h);
                model.addConstr(Y(s, h) <= (X(i, h) + X(j, h)) / 2, name);
                s++;
            }
		}
    }
}
 */


void comb_gurobi_model::set_objective_function(arma::mat &dist) {
    GRBQuadExpr obj = 0;
    for (int h = 0; h < p; h++) {
//        int s = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                obj += dist(i, j) * X(i, h) * X(j, h);
//                s++;
            }
        }
    }
    model.setObjective(obj, GRB_MAXIMIZE);
}

void comb_gurobi_model::optimize(){
	try {
        //std::string file = log_path;
        //auto name = file.substr(0, file.find_last_of("."));
        //model.write(name + ".lp");
        
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}



int comb_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

arma::mat comb_gurobi_model::get_x_solution() {
	arma::mat Xopt(n, p);
	for (int i = 0; i < n; i++) {
		for (int h = 0; h < p; h++) {
			Xopt(i, h) = X(i, h).get(GRB_DoubleAttr_X);
		}
	}
	return Xopt;
}

double comb_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}


