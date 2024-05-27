#include "glover_model.h"

template<class T>
GWMatrix<T>::GWMatrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &GWMatrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T GWMatrix<T>::operator()(size_t row, size_t col) const {
    return data[row*cols+col];
}



std::string glover_model::get_x_variable_name(int i, int h){
	std::ostringstream os;
	os << "x" << i << "_" << h;
	return os.str();
}

std::string glover_model::get_w_variable_name(int i, int h){
	std::ostringstream os;
	os << "W" << i << "_" << h;
	return os.str();
}

std::string glover_model::get_point_constraint_name(int i){
	std::ostringstream os;
	os << "X" << i;
	return os.str();
}

std::string glover_model::get_part_constraint_name(int c, int h){
	std::ostringstream os;
	os << "C" << c << "_" << h;
	return os.str();
}


std::string glover_model::get_lb_constraint_name(int i, int h){
	std::ostringstream os;
	os << "lb_" << i << "_" << h;
	return os.str();
}


std::string glover_model::get_ub_constraint_name(int i, int h){
	std::ostringstream os;
	os << "ub_" << i << "_" << h;
	return os.str();
}



glover_gurobi_model::glover_gurobi_model(GRBEnv *env, int n, int p, int k, arma::mat dist, arma::mat lb, arma::mat ub) : model(*env), X(n,p), W(n,p) {
	this->n = n;
	this->p = p;
	this->k = k;
	this->dist = dist;
	this->lb = lb;
	this->ub = ub;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->W = create_W_variables(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
	this->model.set("OptimalityTol", "1e-4");
    //this->model.set("TimeLimit", "120");
    //this->model.set("Presolve", 1);
}

GWMatrix<GRBVar> glover_gurobi_model::create_X_variables(GRBModel &model) {
    GWMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < p; h++) {
			std::string name = get_x_variable_name(i, h);
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
        }
    }
    return X;
}

GWMatrix<GRBVar> glover_gurobi_model::create_W_variables(GRBModel &model) {
	GWMatrix<GRBVar> W(n, p);
	for (int i = 0; i < n; i++) {
		for (int h = 0; h < p; h++) {
			std::string name = get_w_variable_name(i, h);
			W(i, h) = model.addVar(0.0, GRB_INFINITY, -1, GRB_CONTINUOUS, name);
		}
	}
	return W;
}


void glover_gurobi_model::add_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        std::string name = get_point_constraint_name(i);
        model.addConstr(lhs_sum == 1, name);
    }
}

void glover_gurobi_model::add_part_constraints() {
    //for (int c = 0; c < k; c++) {
	    for (int h = 0; h < p; h++) {
    		int nc = 0;
	        GRBLinExpr lhs_sum = 0;
	        for (int i = 0; i < n; i++)
	        	//if (init_sol(i,c) == 1) {
	            	lhs_sum += X(i, h);
	            //	nc++;
	            //}
	    	//model.addConstr(lhs_sum >= std::floor(n/p), "P" + std::to_string(h));
	        //std::string name = get_part_constraint_name(c, h);
	    	model.addConstr(lhs_sum >= std::floor(n/p));
	    	//model.addConstr(lhs_sum <= nc/(p - 1), name);
	    	//model.addConstr(lhs_sum >= nc/(p + 1) , name);
	    }
    //}
}

/*
void glover_gurobi_model::add_bound_constraints() {
    for (int h = 0; h < p; h++) {
        for (int i = 0; i < n; i++) {
        	GRBQuadExpr lb_i = 0;
            for (int c = 0; c < k; c++) {
            	if (init_sol(i,c) == 1) {
            		model.addConstr(W(i,h) <= ub(i) * X(i, h), get_ub_constraint_name(i, h));
            		lb_i -= lb(i)*(1 - X(i, h));
            	}
            	for (int j = 0; j < n; j++)
            		if (i != j and init_sol(i,c) == 1 and init_sol(j,c) ==1)
	                	lb_i +=  dist(i, j) * X(j, h) / 2;
            }
			model.addConstr(W(i,h) <= lb_i, get_lb_constraint_name(i, h));
		}
    }
}
*/

void glover_gurobi_model::add_bound_constraints() {
	for (int h = 0; h < p; h++) {
		for (int i = 0; i < n; i++) {
			GRBQuadExpr lb_i = 0;
			GRBQuadExpr ub_i = 0;
			for (int j = 0; j < n; j++) {
				for (int c = 0; c < k; c++)
					if (i != j and init_sol(i,c) == 1 and init_sol(j,c) ==1) {
						lb_i +=  (dist(i, j) / 2) * X(j, h) ;
						ub_i +=  (dist(i, j) / 2) * X(i, h) ;
					}
			}
			//model.addConstr(W(i,h) >= lb(i,1)*X(i, h));
			//model.addConstr(W(i,h) <= ub(i,1)*X(i, h));
			//model.addConstr(W(i,h) <= lb_i - lb(i,0)*(1-X(i, h)));
			//model.addConstr(W(i,h) >= lb_i - ub(i,0)*(1-X(i, h)));
			model.addConstr(W(i,h) <= lb_i);
			model.addConstr(W(i,h) <= ub_i);
		}
	}
}

void glover_gurobi_model::optimize(){
	try {
        std::string file = sol_path;
        auto name = file.substr(0, file.find_last_of("."));
        model.write(name + ".lp");
        std::cout << std::endl << std::endl << name << std::endl;

		model.optimize();
		status = model.get(GRB_IntAttr_Status);
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}



int glover_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

arma::mat glover_gurobi_model::get_x_solution() {
	arma::mat Xopt(n, p);
	for (int i = 0; i < n; i++) {
		for (int h = 0; h < p; h++) {
        	std::cout << X(i, h).get(GRB_DoubleAttr_X);
			Xopt(i, h) = X(i, h).get(GRB_DoubleAttr_X);
		}
	}
	return Xopt;
}

double glover_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}


