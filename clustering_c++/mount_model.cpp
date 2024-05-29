#include "mount_model.h"

template<class T>
MMatrix<T>::MMatrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &MMatrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T MMatrix<T>::operator()(size_t row, size_t col) const {
    return data[row*cols+col];
}



std::string mount_model::get_x_variable_name(int c1, int h1, int t){
	std::ostringstream os;
	os << "X" << c1 << "_" << h1 << "_" << t;
	return os.str();
}

std::string mount_model::get_y_variable_name(int c1, int h1, int c2, int h2, int t){
	std::ostringstream os;
	os << "Y" << c1 << "_" << h1 << "_" << c2 << "_" << h2 << "_" << t;
	return os.str();
}


mount_gurobi_model::mount_gurobi_model(GRBEnv *env, int m, std::vector<std::vector<double>> &dist, std::unordered_map<int, std::unordered_map<int, arma::mat>> &sol_cls) : model(*env), X(k*p,p), Y(m,p) {
	this->m = m;
	this->dist = dist;
	this->sol_cls = sol_cls;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->Y = create_Y_variables(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
	this->model.set("OptimalityTol", "1e-4");
    //this->model.set("TimeLimit", "120");
    //this->model.set("Presolve", 1);
}

MMatrix<GRBVar> mount_gurobi_model::create_X_variables(GRBModel &model) {
	MMatrix<GRBVar> X(k*p, p);
	int s = 0;
	for (int c1 = 0; c1 < k; c1++) {
		for (int h1=0; h1 < p; ++h1) {
			for (int t=0; t < p; ++t) {
				std::string name = get_x_variable_name(c1, h1, t);
				X(s, t) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
			}
			s++;
		}
	}
	return X;
}

MMatrix<GRBVar> mount_gurobi_model::create_Y_variables(GRBModel &model) {
    MMatrix<GRBVar> Y(m, p);
    int s = 0;
    for (int c1 = 0; c1 < k-1; c1++) {
    	for (int h1=0; h1 < p-1; ++h1) {
    		for (int c2 = c1+1; c2 < k; c2++) {
    			if (c1!=c2) {
    				for (int h2=h1+1; h2 < p; ++h2) {
    					double obj = 0;
    					for (int i = 0; i < sol_cls[c1][h1].n_rows; i++)
    						for (int j = 0; j < sol_cls[c2][h2].n_rows; j++)
    							obj += dist[sol_cls[c1][h1](i,0) - 1][sol_cls[c2][h2](j,0) - 1];
    					for (int t=0; t < p; ++t) {
    						std::string name = get_y_variable_name(c1, h1, c2, h2, t);
    						Y(s, t) = model.addVar(0.0, 1, -obj, GRB_BINARY, name);
    					}
    					s++;
    				}
				}
			}
		}
	}
    return Y;
}

void mount_gurobi_model::add_point_constraints() {
	int s = 0;
	for (int c1 = 0; c1 < k; c1++) {
		for (int h1=0; h1 < p; ++h1) {
        	GRBLinExpr lhs_sum = 0;
			for (int t=0; t < p; ++t)
            	lhs_sum += X(s, t);
			model.addConstr(lhs_sum == 1);
			s++;
		}
	}
}

void mount_gurobi_model::add_cls_constraints() {
	for (int t=0; t < p; ++t) {
		int s = 0;
		for (int c1 = 0; c1 < k; c1++) {
        	GRBLinExpr lhs_sum = 0;
			for (int h1=0; h1 < p; ++h1) {
            	lhs_sum += X(s, t);
				s++;
			}
    		model.addConstr(lhs_sum == 1);
		}
	}
}

void mount_gurobi_model::add_edge_constraints() {
	for (int t = 0; t < p; t++) {
		int s = 0;
		int s1 = 0;
		for (int c1 = 0; c1 < k-1; c1++) {
			for (int h1=0; h1 < p-1; ++h1) {
				int s2 = 0;
				for (int c2 = c1+1; c2 < k; c2++) {
					if (c1!=c2) {
						for (int h2=h1+1; h2 < p; ++h2) {
							model.addConstr(Y(s, t) <= X(s1, t));
							model.addConstr(Y(s, t) <= X(s2, t));
							model.addConstr(Y(s, t) >= X(s1, t) + X(s2, t)  - 1 );
							s2++;
							s++;
						}
					}
				}
				s1++;
			}
		}
	}
}


void mount_gurobi_model::optimize(){
	try {
        std::string file = sol_path;
        auto name = file.substr(0, file.find_last_of("."));
        //model.write(name + ".lp");
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}



int mount_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

std::map<int, arma::vec> mount_gurobi_model::get_x_solution() {
	std::map<int, arma::vec> Xopt;
	for (int t=0; t < p; ++t) {
		Xopt[t] = arma::vec(k) - 1;
		int s = 0;
		for (int c1 = 0; c1 < k; c1++) {
			for (int h1 = 0; h1 < p; ++h1) {
				if (X(s, t).get(GRB_DoubleAttr_X) > 0.8)
					Xopt[t](c1) = h1;
				s++;
			}
		}
	}
	return Xopt;
}

double mount_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}


