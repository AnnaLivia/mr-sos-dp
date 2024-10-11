#include "cluster_model.h"

template<class T>
CLMatrix<T>::CLMatrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &CLMatrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T CLMatrix<T>::operator()(size_t row, size_t col) const {
	return data[row*cols+col];
}

cluster_gurobi_model::cluster_gurobi_model(GRBEnv *env, int n, int p, int k, int size_y, std::vector<std::vector<double>> dist, std::vector<std::vector<int>> cls_points) : model(*env), X(n,p), Y(size_y,p), Z() {
	this->n = n;
	this->p = p;
	this->k = k;
	this->size_y = size_y;
	this->dist = dist;
	this->cls_points = cls_points;
	this->env = env;
	this->X = create_X_variables(this->model);
	this->Y = create_Y_variables(this->model);
	this->Z = create_Z_variable(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
    this->model.set("TimeLimit", "1800");
    //this->model.set("Presolve", 1);
	model.set(GRB_StringParam_LogFile, result_path + "_GRB_cls_log.txt");
}


CLMatrix<GRBVar> cluster_gurobi_model::create_X_variables(GRBModel &model) {
    CLMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < p; h++) {
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, "X_" + std::to_string(i)
                + "_" + std::to_string(h));
        }
    }
    return X;
}

CLMatrix<GRBVar> cluster_gurobi_model::create_Y_variables(GRBModel &model) {
    CLMatrix<GRBVar> Y(n*(n-1)/2, p);
    int s = 0;
    for (int c = 0; c < k; c++) {
        for (int i: cls_points[c]) {
        	for (int j: cls_points[c]) {
        		if (i<j) {
    				for (int h = 0; h < p; h++) {
    					int points = floor(cls_points[c].size()/p);
    					if (h < cls_points[c].size() % p)
    						points += 1;
                		Y(s, h) = model.addVar(0.0, 1.0, - dist[i][j]/points, GRB_BINARY, "Y_" + std::to_string(i)
                		+ "_" + std::to_string(j) + "_" + std::to_string(h));
            		}
                	s++;
                }
            }
        }
    }
    return Y;
}

GRBVar cluster_gurobi_model::create_Z_variable(GRBModel &model) {
	GRBVar Z = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Z");
	return Z;
}

void cluster_gurobi_model::add_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        model.addConstr(lhs_sum == 1, "P" + std::to_string(i));
    }
}

void cluster_gurobi_model::add_part_constraints() {
    for (int h = 0; h < p; h++) {
    	for (int c = 0; c < k; c++) {
        	GRBLinExpr lhs_sum = 0;
        	for (int i: cls_points[c])
            	lhs_sum += X(i, h);
    		int points = floor(cls_points[c].size()/p);
    		if (h < cls_points[c].size() % p)
    			points += 1;
    		model.addConstr(lhs_sum == points, "C" + std::to_string(c) + "P" + std::to_string(h));
    	}
    }
}

void cluster_gurobi_model::add_edge_constraints() {
    int s = 0;
    for (int c = 0; c < k; c++) {
        for (int i: cls_points[c])
        	for (int j: cls_points[c])
        		if (i<j) {
        			GRBLinExpr lhs_sum = 0;
    				for (int h = 0; h < p; h++) {
    					lhs_sum += Y(s,h);
	            		model.addConstr(Y(s, h) <= X(i, h));
	            		model.addConstr(Y(s, h) <= X(j, h));
	            		model.addConstr(Y(s, h) >= X(i, h) + X(j, h) - 1);
	            	}
    				model.addConstr(lhs_sum <= 1);
                	s++;
                }
    }

	/*
    for (int h = 0; h < p; h++) {
        GRBLinExpr lhs_sum = 0;
    	int s = 0;
    	for (int c = 0; c < k; c++) {
    		int points = floor(cls_points[c].size()/p);
    		if (h < cls_points[c].size() % p)
    			points += 1;
        	for (int i: cls_points[c])
        		for (int j: cls_points[c])
        			if (i<j) {
    					lhs_sum += Y(s,h)*dist[i][j]/points;
    					s++;
    				}
        }
    	model.addConstr(Z <= lhs_sum);
    }
    */

}

void cluster_gurobi_model::optimize(){
	try {
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
		model.write(result_path + "_GRB_cls.lp");
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}

int cluster_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

void cluster_gurobi_model::get_x_solution(std::vector<std::vector<int>> &sol) {
	for (int h = 0; h < p; h++) {
		sol[h].reserve(n);
		for (int i = 0; i < n; i++) {
        	if (X(i, h).get(GRB_DoubleAttr_X) > 0.8) {
        		sol[h].push_back(i);
        	}
		}
	}
}

double cluster_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}

double cluster_gurobi_model::get_gap(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_MIPGap);
}


//Model per single cluster
gurobi_model::gurobi_model(GRBEnv *env, int n, int p, int k, std::vector<std::vector<double>> dist, std::vector<int> cls) : model(*env), X(n,p), Y(n*(n-1)/2,p) {
	this->n = n;
	this->p = p;
	this->k = k;
	this->dist = dist;
	this->cls = cls;
	this->env = env;
	this->X = create_Xcls_variables(this->model);
	this->Y = create_Ycls_variables(this->model);
    this->model.set("OutputFlag", "1");
	this->model.set("Threads", "4");
    this->model.set("TimeLimit", "300");
    //this->model.set("Presolve", 1);
	model.set(GRB_StringParam_LogFile, result_path + "_GRB_cls_log.txt");
}


CLMatrix<GRBVar> gurobi_model::create_Xcls_variables(GRBModel &model) {
    CLMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < p; h++) {
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, "X_" + std::to_string(i)
                + "_" + std::to_string(h));
        }
    }
    return X;
}

CLMatrix<GRBVar> gurobi_model::create_Ycls_variables(GRBModel &model) {
    CLMatrix<GRBVar> Y(n*(n-1)/2, p);
    int s = 0;
    for (int i = 0; i < n; i++) {
    	for (int j = i+1; j < n; j++) {
        	GRBLinExpr lhs_sum = 0;
    		for (int h = 0; h < p; h++) {
    			int points = floor(n/p);
    			if (h < n % p)
    				points += 1;
                Y(s, h) = model.addVar(0.0, 1.0, -dist[cls[i]][cls[j]]/points, GRB_BINARY, "Y_" + std::to_string(i)
                		+ "_" + std::to_string(j) + "_" + std::to_string(h));
            	lhs_sum += Y(s, h);
            }
        	model.addConstr(lhs_sum <= 1);
            s++;
        }
    }
    return Y;
}

void gurobi_model::add_cls_point_constraints() {
    for (int i = 0; i < n; i++) {
        GRBLinExpr lhs_sum = 0;
        for (int h = 0; h < p; h++)
            lhs_sum += X(i, h);
        model.addConstr(lhs_sum == 1);
    }
}

void gurobi_model::add_cls_part_constraints() {
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

void gurobi_model::add_cls_edge_constraints() {
    int s = 0;
	for (int i = 0; i < n; i++) {
		for (int j = i+1; j < n; j++) {
        	GRBLinExpr lhs_sum = 0;
    		for (int h = 0; h < p; h++) {
	            model.addConstr(Y(s, h) <= X(i, h));
	            model.addConstr(Y(s, h) <= X(j, h));
	            model.addConstr(Y(s, h) >= X(i, h) + X(j, h) - 1);
	        }
	        s++;
	    }
    }

    for (int h = 0; h < p; h++) {
    	s = 0;
		int points = floor(n/p);
		if (h < n % p)
			points += 1;
        std::vector<GRBLinExpr> lhs_sum(n);
		for (int i = 0; i < n; i++) {
			for (int j = i+1; j < n; j++) {
	            lhs_sum[i] += Y(s, h);
	            lhs_sum[j] += Y(s, h);
	        	s++;
	        }
	    }
		for (int i = 0; i < n; i++)
	    	model.addConstr(lhs_sum[i] == (points -1)*X(i,h));
    }
}

void gurobi_model::optimize_cls(){
	try {
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
		model.write(result_path + "_GRB_cls.lp");
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}

void gurobi_model::get_cls_solution(std::vector<std::vector<int>> &sol) {
	for (int h = 0; h < p; h++) {
		sol[h].reserve(n);
		for (int i = 0; i < n; i++)
        	if (X(i, h).get(GRB_DoubleAttr_X) > 0.8)
        		sol[h].push_back(cls[i]);
	}
}

double gurobi_model::get_cls_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}

double gurobi_model::get_cls_gap(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_MIPGap);
}