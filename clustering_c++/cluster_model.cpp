#include "cluster_model.h"

template<class T>
GRBMatrix<T>::GRBMatrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &GRBMatrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T GRBMatrix<T>::operator()(size_t row, size_t col) const {
    return data[row*cols+col];
}



std::string cluster_model::get_x_variable_name(int i, int h){
	std::ostringstream os;
	os << "x" << i << "_" << h;
	return os.str();
}

std::string cluster_model::get_y_variable_name(int i, int j, int h){
	std::ostringstream os;
	os << "y" << i << "_" << j << "_" << h;
	return os.str();
}

std::string cluster_model::get_point_constraint_name(int i){
	std::ostringstream os;
	os << "X" << i;
	return os.str();
}

std::string cluster_model::get_part_constraint_name(int c, int h){
	std::ostringstream os;
	os << "K" << c << "P" << h;
	return os.str();
}


std::string cluster_model::get_edge_constraint_name(int i, int j, int h){
	std::ostringstream os;
	os << "C" << i << "_" << j << "_" << h;
	return os.str();
}



cluster_gurobi_model::cluster_gurobi_model(GRBEnv *env, int n, int p, int k, arma::mat dist) : model(*env), X(n,p), Y(n*(n-1)/2,p), Z(p,1) {
	this->n = n;
	this->p = p;
	this->k = k;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->Y = create_Y_variables(this->model, dist);
	this->Z = create_Z_variables(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
	this->model.set("OptimalityTol", "1e-5");
    this->model.set("TimeLimit", "300");
    //this->model.set("Presolve", 1);
}

GRBMatrix<GRBVar> cluster_gurobi_model::create_X_variables(GRBModel &model) {
    GRBMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < p; h++) {
			std::string name = get_x_variable_name(i, h);
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
        }
    }
    return X;
}

GRBMatrix<GRBVar> cluster_gurobi_model::create_Y_variables(GRBModel &model, arma::mat dist) {
    GRBMatrix<GRBVar> Y(n*(n-1)/2, p);
    for (int h = 0; h < p; h++) {
        int s = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
            	std::string name = get_y_variable_name(i, j, h);
                Y(s, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
            	/*
            	double obj = 0;
            	for (int c = 0; c < k; c++)
            		if (init_sol(i,c) == 1 and init_sol(j,c) ==1)
            			obj = dist(i, j) * Y(s, h);
            	 */
                s++;
            }
        }
    }
    return Y;
}

GRBMatrix<GRBVar> cluster_gurobi_model::create_Z_variables(GRBModel &model) {
	GRBMatrix<GRBVar> Z(p,1);
	for (int h = 0; h < p; h++)
		Z(h,0) = model.addVar(0.0, GRB_INFINITY, -1.0, GRB_CONTINUOUS, "z_" + std::to_string(h));
	return Z;
}



void cluster_gurobi_model::add_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        std::string name = get_point_constraint_name(i);
        model.addConstr(lhs_sum == 1, name);
    }
}

void cluster_gurobi_model::add_part_constraints() {
    //for (int c = 0; c < k; c++) {
	    for (int h = 0; h < p; h++) {
    		int nc = 0;
	        GRBLinExpr lhs_sum = 0;
	        for (int i = 0; i < n; i++) {
	        	//if (init_sol(i,c) == 1) {
	            	lhs_sum += X(i, h);
	            	nc++;
	            }
	    	model.addConstr(lhs_sum >= std::floor(n/p), "P" + std::to_string(h));
	        //std::string name = get_part_constraint_name(c, h);
	    	//model.addConstr(lhs_sum >= std::floor(nc/p), name);
	    	//model.addConstr(lhs_sum <= nc/(p - 1), name);
	    	//model.addConstr(lhs_sum >= nc/(p + 1) , name);
	    }
    //}
}


void cluster_gurobi_model::add_edge_constraints() {
    for (int h = 0; h < p; h++) {
        int s = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
            	for (int c = 0; c < k; c++)
            		//if (init_sol(i,c) == 1 and init_sol(j,c) ==1) {
	                	model.addConstr(Y(s, h) <= X(i, h));
	                	model.addConstr(Y(s, h) <= X(j, h));
	                	std::string name = get_edge_constraint_name(i, j, h);
	                	model.addConstr(Y(s, h) >= X(i, h) + X(j, h)  -1 , name);
                	//}
                s++;
            }
		}
    }
}

void cluster_gurobi_model::add_min_constraints(arma::mat dist) {
	for (int h = 0; h < p; h++) {
    	GRBQuadExpr con = 0;
		int s = 0;
		for (int i = 0; i < n-1; i++) {
			for (int j = i+1; j < n; j++) {
    			//for (int c = 0; c < k; c++) {
					//if (init_sol(i,c) == 1 and init_sol(j,c) ==1)
						con += dist(i, j) * Y(s, h);
				//}
				s++;
			}
		}
		model.addConstr(Z(h,0) <= con, "P_" + std::to_string(h));
	}
}


void cluster_gurobi_model::optimize(){
	try {
        std::string file = sol_path;
        auto name = file.substr(0, file.find_last_of("."));
        model.write(name + ".lp");

		model.optimize();
		status = model.get(GRB_IntAttr_Status);
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}



int cluster_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

arma::mat cluster_gurobi_model::get_x_solution() {
	arma::mat Xopt(n, p);
	for (int i = 0; i < n; i++) {
		for (int h = 0; h < p; h++) {
        	std::cout << X(i, h).get(GRB_DoubleAttr_X);
			Xopt(i, h) = X(i, h).get(GRB_DoubleAttr_X);
		}
	}
	return Xopt;
}

double cluster_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}


