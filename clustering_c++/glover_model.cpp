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

glover_gurobi_model::glover_gurobi_model(GRBEnv *env, int n, int p, int k, std::vector<std::vector<double>> dist, std::vector<std::vector<int>> cls_points) : model(*env), X(n,p), W(n,p), Z() {
	this->n = n;
	this->p = p;
	this->k = k;
	this->dist = dist;
	this->cls_points = cls_points;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->W = create_W_variables(this->model);
	this->Z = create_Z_variable(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
	this->model.set("OptimalityTol", "1e-4");
    this->model.set("TimeLimit", "300");
    //this->model.set("Presolve", 1);
	model.set(GRB_StringParam_LogFile, result_path + "_GW_log.txt");
}

GWMatrix<GRBVar> glover_gurobi_model::create_X_variables(GRBModel &model) {
    GWMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++)
        for (int h = 0; h < p; h++)
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, "X" + std::to_string(i) + "_" + std::to_string(h));
    return X;
}

GWMatrix<GRBVar> glover_gurobi_model::create_W_variables(GRBModel &model) {
	GWMatrix<GRBVar> W(n, p);
	for (int i = 0; i < n; i++)
		for (int h = 0; h < p; h++)
			W(i, h) = model.addVar(0.0, GRB_INFINITY, -1.0, GRB_CONTINUOUS, "W" + std::to_string(i) + "_" + std::to_string(h));
	return W;
}

GRBVar glover_gurobi_model::create_Z_variable(GRBModel &model) {
	GRBVar Z = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Z");
	return Z;
}

void glover_gurobi_model::add_point_constraints() {
	for (int i = 0; i < n; i++) {
		GRBLinExpr lhs_sum = 0;
		for (int h = 0; h < p; h++)
			lhs_sum += X(i, h);
		model.addConstr(lhs_sum == 1, "P" + std::to_string(i));
	}
}

void glover_gurobi_model::add_part_constraints() {
	for (int h = 0; h < p; h++) {
		for (int c = 0; c < k; c++) {
			bool first = true;
			GRBLinExpr lhs_sum = 0;
			for (int i: cls_points[c]) {
				if (h==0 and first) {
					model.addConstr(X(i, h) == 1);
					first = false;
				}
				lhs_sum += X(i, h);
			}
			int points = floor(cls_points[c].size()/p);
			if (h < cls_points[c].size() % p)
				points += 1;
			model.addConstr(lhs_sum == points, "C" + std::to_string(c) + "P" + std::to_string(h));
		}
	}
}

/*
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
		for (int c = 0; c < k; c++) {
			int nc = cls_points[c].size();
			int points = floor(nc/p);
			if (h < nc % p)
				points += 1;
			for (int i: cls_points[c]) {
				GRBLinExpr ub0 = 0;
				std::vector<double> dist_vec(nc - 1);
				for (int j: cls_points[c]) {
					if (i != j) {
						ub0 +=  dist[i][j]/2 * X(j, h);
						dist_vec.push_back(dist[i][j]/2);
					}
				}

				std::sort(dist_vec.begin(), dist_vec.end(), std::greater<double>());
				double ub1 = 0;
				for (int t = 0; t < points-1; t++)
					ub1 += dist_vec[t];
				model.addConstr(W(i,h) <= X(i, h)*ub1/points, "Cls" + std::to_string(c) + "_UB1" + std::to_string(i) + "_P" + std::to_string(h));
				model.addConstr(W(i,h) <= ub0/points, "Cls" + std::to_string(c) + "_UB0" + std::to_string(i) + "_P" + std::to_string(h));

				std::sort(dist_vec.begin(), dist_vec.end());
				double lb1 = 0;
				for (int t = 0; t < points-1; t++)
					lb1 += dist_vec[t];
				model.addConstr(W(i,h) >= X(i, h)*lb1/points, "Cls" + std::to_string(c) + "_LB1" + std::to_string(i) + "_P" + std::to_string(h));
				//model.addConstr(W(i,h) >= lb(i,1)*X(i, h));
				//model.addConstr(W(i,h) <= ub(i,1)*X(i, h));
				//model.addConstr(W(i,h) <= lb_i - lb(i,0)*(1-X(i, h)));
				//model.addConstr(W(i,h) >= lb_i - ub(i,0)*(1-X(i, h)));
			}
		}
	}
}


void glover_gurobi_model::optimize(){
	try {
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
		model.write(result_path + "_GRB_gw.lp");
	} catch (GRBException &e) {
		std::cout << "Error code = " << e.getErrorCode() << std::endl;
		std::cout << e.getMessage() << std::endl;
	}
}

int glover_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

void glover_gurobi_model::get_x_solution(std::vector<std::vector<int>> &sol) {
	for (int h = 0; h < p; h++) {
		sol[h].reserve(n);
		for (int i = 0; i < n; i++) {
			if (X(i, h).get(GRB_DoubleAttr_X) > 0.8) {
				sol[h].push_back(i);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		std::cout << "Point" << i << ":  ";
		for (int h = 0; h < p; h++) {
			if (W(i, h).get(GRB_DoubleAttr_X) == 0)
				std::cout << " NO w" << h;
			else
				std::cout << " P" << h << "i" << W(i, h).get(GRB_DoubleAttr_X) << std::endl;
		}
	}
}

double glover_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}

double glover_gurobi_model::get_gap(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_MIPGap);
}



//Model per cluster
glover_cls_model::glover_cls_model(GRBEnv *env, int n, int p, int k, std::vector<std::vector<double>> dist, std::vector<int> cls) : model(*env), X(n,p), W(n,p) {
	this->n = n;
	this->p = p;
	this->k = k;
	this->dist = dist;
	this->cls = cls;
	this->env = env;
	this->X = create_Xcls_variables(this->model);
	this->W = create_Wcls_variables(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
	this->model.set("OptimalityTol", "1e-4");
    this->model.set("TimeLimit", "300");
    //this->model.set("Presolve", 1);
	model.set(GRB_StringParam_LogFile, result_path + "_GW_log.txt");
}


GWMatrix<GRBVar> glover_cls_model::create_Xcls_variables(GRBModel &model) {
    GWMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++)
        for (int h = 0; h < p; h++)
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, "X" + std::to_string(i) + "_" + std::to_string(h));
    return X;
}

GWMatrix<GRBVar> glover_cls_model::create_Wcls_variables(GRBModel &model) {
	GWMatrix<GRBVar> W(n, p);
	for (int i = 0; i < n; i++)
		for (int h = 0; h < p; h++)
			W(i, h) = model.addVar(0.0, GRB_INFINITY, -1.0, GRB_CONTINUOUS, "W" + std::to_string(i) + "_" + std::to_string(h));
	return W;
}

void glover_cls_model::add_cls_point_constraints() {
	for (int i = 0; i < n; i++) {
		GRBLinExpr lhs_sum = 0;
		for (int h = 0; h < p; h++)
			lhs_sum += X(i, h);
		model.addConstr(lhs_sum == 1, "P" + std::to_string(i));
	}
}

void glover_cls_model::add_cls_part_constraints() {
	for (int h = 0; h < p; h++) {
		GRBLinExpr lhs_sum = 0;
		for (int i = 0; i < n; i++) {
			if (h==0 and i==0)
				model.addConstr(X(i, h) == 1);
			lhs_sum += X(i, h);
		}
		int points = floor(n/p);
		if (h < n % p)
			points += 1;
		model.addConstr(lhs_sum == points, "P" + std::to_string(h));
	}
}

void glover_cls_model::add_cls_bound_constraints() {
	for (int h = 0; h < p; h++) {
		int points = floor(n/p);
		if (h < n % p)
			points += 1;
		for (int i = 0; i < n; i++) {
			GRBLinExpr ub0 = 0;
			std::vector<double> dist_vec(n - 1);
			for (int j = i+1; j < n; j++) {
				ub0 +=  dist[cls[i]][cls[j]]/2 * X(j, h);
				dist_vec.push_back(dist[cls[i]][cls[j]]/2);
			}

			std::sort(dist_vec.begin(), dist_vec.end(), std::greater<double>());
			double ub1 = 0;
			for (int t = 0; t < points-1; t++)
				ub1 += dist_vec[t];
			model.addConstr(W(i,h) <= X(i, h)*ub1/points, "UB1" + std::to_string(i) + "_P" + std::to_string(h));
			model.addConstr(W(i,h) <= ub0/points, "UB0" + std::to_string(i) + "_P" + std::to_string(h));

			std::sort(dist_vec.begin(), dist_vec.end());
			double lb1 = 0;
			for (int t = 0; t < points-1; t++)
				lb1 += dist_vec[t];
			model.addConstr(W(i,h) >= X(i, h)*lb1/points, "LB1" + std::to_string(i) + "_P" + std::to_string(h));
		}
	}
}


void glover_cls_model::optimize_cls(){
	try {
		model.write(result_path + "_GRB_gw.lp");
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
	} catch (GRBException &e) {
		std::cout << "Error code = " << e.getErrorCode() << std::endl;
		std::cout << e.getMessage() << std::endl;
	}
}

void glover_cls_model::get_cls_solution(std::vector<std::vector<int>> &sol) {
	for (int h = 0; h < p; h++) {
		sol[h].reserve(n);
		for (int i = 0; i < n; i++) {
			if (X(i, h).get(GRB_DoubleAttr_X) > 0.8) {
				sol[h].push_back(cls[i]);
			}
		}
	}
	for (int i = 0; i < n; i++) {
		std::cout << "Point" << i << ":  ";
		for (int h = 0; h < p; h++) {
			if (W(i, h).get(GRB_DoubleAttr_X) == 0)
				std::cout << " NO w" << h;
			else
				std::cout << " P" << h << "i" << W(i, h).get(GRB_DoubleAttr_X) << std::endl;
		}
	}
}

double glover_cls_model::get_cls_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}

double glover_cls_model::get_cls_gap(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_MIPGap);
}