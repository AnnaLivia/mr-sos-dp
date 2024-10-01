#include "comb_model.h"

template<class T>
CMatrix<T>::CMatrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &CMatrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T CMatrix<T>::operator()(size_t row, size_t col) const {
	return data[row*cols+col];
}


std::string comb_model::get_x_variable_name(int i, int h){
	std::ostringstream os;
	os << "x" << i << "_" << h;
	return os.str();
}

std::string comb_model::get_y_variable_name(int l){
	std::ostringstream os;
	os << "y" << l;
	return os.str();
}

std::string comb_model::get_point_constraint_name(int i){
	std::ostringstream os;
	os << "X" << i;
	return os.str();
}

std::string comb_model::get_part_constraint_name(int h){
	std::ostringstream os;
	os << "P" << h;
	return os.str();
}


std::string comb_model::get_edge_constraint_name(int i, int j, int h){
	std::ostringstream os;
	os << "C" << i << "_" << j << "_" << h;
	return os.str();
}



comb_gurobi_model::comb_gurobi_model(GRBEnv *env, int n, int p, int d, std::vector<int> cls_points) : model(*env), X(n,p), Y(1,d), Z()

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
    this->model.set("TimeLimit", "180");
    //this->model.set("Presolve", 1);
	model.set(GRB_StringParam_LogFile, result_path + "_GRB_comb_log.txt");
}


CMatrix<GRBVar> comb_gurobi_model::create_X_variables(GRBModel &model) {
    CMatrix<GRBVar> X(n, p);
    for (int i = 0; i < n; i++) {
        for (int h = 0; h < p; h++) {
			std::string name = get_x_variable_name(i, h);
            X(i, h) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
        }
    }
    return X;
}

CMatrix<GRBVar> comb_gurobi_model::create_Y_variables(GRBModel &model) {
    CMatrix<GRBVar> Y(1,d);
    for (int l = 0; l < d; l++) {
        std::string name = get_y_variable_name(l);
        Y(0, l) = model.addVar(0.0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, name);
    }
    return Y;
}

GRBVar comb_gurobi_model::create_Z_variable(GRBModel &model) {
	GRBVar Z = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Z");
	return Z;
}

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
    	int points = floor(n/p);
    	if (h < n % p)
    		points += 1;
    	model.addConstr(lhs_sum == points, name);
    }
}

void comb_gurobi_model::add_edge_constraints() {
    for (int h = 0; h < p; h++) {
        int s = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i+1; j < n; j++) {
                model.addConstr(Y(s, h) <= X(i, h));
                model.addConstr(Y(s, h) <= X(j, h));
                std::string name = get_edge_constraint_name(i, j, h);
                model.addConstr(Y(s, h) >= X(i, h) + X(j, h)  -1 , name);
                s++;
            }
		}
    }
}

void comb_gurobi_model::add_dev_constraints(std::vector<std::vector<double>> &data, std::vector<double> &center) {
    for (int h = 0; h < p; h++) {
    	int points = floor(n/p);
    	if (h < n % p)
    		points += 1;
    	for (int l = 0; l < d; l++) {
        	GRBLinExpr lhs_sum = 0;
        	std::vector<double> diff(n);
        	for (int i = 0; i < n; i++) {
            	lhs_sum += data[cls_points[i]][l]*X(i, h)/points;
            	diff.push_back(data[cls_points[i]][l]/points - center[l]);
            }
    		std::sort(diff.begin(), diff.end(), [](double a, double b) {
				return std::abs(a) < std::abs(b);  // Comparator to sort by absolute values
			});
    		double abs_sum = 0;
    		for (int i = 0; i < points; ++i)
        		abs_sum += diff[i];
            model.addConstr(Y(0,l) >= lhs_sum - center[l]);
            model.addConstr(Y(0,l) >= - lhs_sum + center[l]);
            model.addConstr(Y(0,l) >= abs_sum);
		}
    }

	/*

	for (int h = 0; h < p; h++) {
		int points = floor(n/p);
		if (h < n % p)
			points += 1;
		for (int l = 0; l < d; l++) {
			GRBLinExpr lhs_sum = 0;
			for (int i = 0; i < n; i++)
				lhs_sum += data[cls_points[i]][l]*X(i, h)/points;
			model.addConstr(Y(0,l) >= lhs_sum - center[l]);
			model.addConstr(Y(0,l) >= - lhs_sum + center[l]);
		}
	}
	*/
}

void comb_gurobi_model::set_objective_function(std::vector<int> &cls_points, std::vector<std::vector<double>> &dist) {
    GRBQuadExpr obj = 0;
    for (int h = 0; h < p; h++) {
        int s = 0;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                obj += dist[cls_points[i]][cls_points[j]] * Y(s, h);
                s++;
            }
        }
    }
    model.setObjective(obj, GRB_MAXIMIZE);
}

void comb_gurobi_model::optimize(){
	try {
		model.optimize();

		/*
		// Get the number of constraints and variables
		int numConstrs = model.get(GRB_IntAttr_NumConstrs) - p*d;
		int numVars = model.get(GRB_IntAttr_NumVars) - 2*d;

		// Retrieve all variables and constraints
		GRBVar* vars = model.getVars();
		GRBConstr* constrs = model.getConstrs();

		std::ofstream f;
		f.open("matrix.txt");
		// Loop over constraints and variables to extract the constraint matrix (A matrix)
		for (int i = 0; i < numConstrs; i++) {
			// For each constraint, loop over all variables
			for (int j = 0; j < numVars; j++) {
				double coef = model.getCoeff(constrs[i], vars[j]); // Get coefficient
				f << coef << " ";
			}
			f << "\n";
		}

		f.close();

		// For each constraint, loop over all variables
		GRBVar* vars = model.getVars();
		int numVars = model.get(GRB_IntAttr_NumVars);
		for (int j = 0; j < numVars; j++) {
			std::cout << vars[j].get(GRB_StringAttr_VarName) << ":" << vars[j].get(GRB_DoubleAttr_X) << "\n";
		}
		*/

		status = model.get(GRB_IntAttr_Status);
		model.write(result_path + "_GRB_comb.lp");
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}

int comb_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

std::vector<std::vector<int>> comb_gurobi_model::get_x_solution() {
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

double comb_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}

double comb_gurobi_model::get_gap(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_MIPGap);
}

