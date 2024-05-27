#include "part_model.h"

template<class T>
GRB_Matrix<T>::GRB_Matrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &GRB_Matrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T GRB_Matrix<T>::operator()(size_t row, size_t col) const {
    return data[row*cols+col];
}

part_gurobi_model::part_gurobi_model(GRBEnv *env, int n, int p, int c, arma::mat dist) : model(*env), X(n,p), Y(n*(n-1)/2,p), Z(p,1) {
	this->n = n;
	this->p = p;
	this->c = c;
	this->dist = dist;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->Y = create_Y_variables(this->model);
	this->Z = create_Z_variables(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
	this->model.set("OptimalityTol", "1e-5");
    this->model.set("TimeLimit", "120");
    //this->model.set("Presolve", 1);
}

GRB_Matrix<GRBVar> part_gurobi_model::create_X_variables(GRBModel &model) {
    GRB_Matrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++)
        for (int h = 0; h < p; h++)
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_CONTINUOUS, "X_" + std::to_string(i));
    return X;
}

GRB_Matrix<GRBVar> part_gurobi_model::create_Y_variables(GRBModel &model) {
    GRB_Matrix<GRBVar> Y(n*(n-1)/2, p);
    for (int h = 0; h < p; h++) {
        int s = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                Y(s, h) = model.addVar(0.0, 1, -dist(i,j), GRB_BINARY, "Y_" + std::to_string(i)
                + "_" + std::to_string(j) + "_" + std::to_string(h));
                s++;
            }
        }
    }
    return Y;
}

GRB_Matrix<GRBVar> part_gurobi_model::create_Z_variables(GRBModel &model) {
	GRB_Matrix<GRBVar> Z(1,1);
	//for (int h = 0; h < p; h++)
	//	Z(h,0) = model.addVar(0.0, GRB_INFINITY,0.0, GRB_CONTINUOUS, "Z_" + std::to_string(h));
	Z(0,0) = model.addVar(0.0, GRB_INFINITY,0.0, GRB_CONTINUOUS, "Z");
	return Z;
}



void part_gurobi_model::add_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        model.addConstr(lhs_sum == 1, "X" + std::to_string(i));
    }
}

void part_gurobi_model::add_part_constraints() {
	int k = init_sol.n_cols;
    for (int h = 0; h < p; h++) {
        GRBLinExpr lhs_sum = 0;
        for (int i = 0; i < n; i++)
	        lhs_sum += X(i, h);
	    model.addConstr(lhs_sum >= std::floor(n/p), "P" + std::to_string(h));
	    //model.addConstr(lhs_sum <= nc/(p - 1), name);
	    //model.addConstr(lhs_sum >= nc/(p + 1) , name);
    }
}


void part_gurobi_model::add_edge_constraints() {
    for (int h = 0; h < p; h++) {
        int s = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
	            model.addConstr(Y(s, h) <= X(i, h), "YS" + std::to_string(s) + "X" + std::to_string(i) + "P" + std::to_string(h));
	            model.addConstr(Y(s, h) <= X(j, h), "YS" + std::to_string(s) + "X" + std::to_string(j) + "P" + std::to_string(h));
	            model.addConstr(Y(s, h) >= X(i, h) + X(j, h)  -1 , "Y" + std::to_string(i) + "_" + std::to_string(j));
                s++;
            }
		}
    }
}

void part_gurobi_model::add_min_constraints() {
	for (int h = 0; h < p; h++) {
    	GRBQuadExpr con = 0;
		int s = 0;
		for (int i = 0; i < n-1; i++) {
			for (int j = i+1; j < n; j++) {
    			con += dist(i, j) * Y(s, h);
				s++;
			}
		}
		model.addConstr(Z(0,0) <= con, "Z_" + std::to_string(h));
	}
}


void part_gurobi_model::optimize(){
	try {
        std::string file = sol_path;
        auto name = file.substr(0, file.find_last_of("."));
        model.write(name + "c_" + std::to_string(c) + ".lp");

		model.optimize();
		status = model.get(GRB_IntAttr_Status);
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}

arma::mat part_gurobi_model::get_x_solution() {
	arma::mat Xopt(n, p);
	for (int i = 0; i < n; i++) {
		for (int h = 0; h < p; h++) {
        	std::cout << X(i, h).get(GRB_DoubleAttr_X);
			Xopt(i, h) = X(i, h).get(GRB_DoubleAttr_X);
		}
	}
	return Xopt;
}

double part_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}


