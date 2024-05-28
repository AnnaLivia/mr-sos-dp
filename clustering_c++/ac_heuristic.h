#ifndef CLUSTERING_AC_HEURISTICS_H
#define CLUSTERING_AC_HEURISTICS_H

#include <unordered_map>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"

typedef struct UserConstraints {

	std::vector<std::pair<int,int>> ml_pairs, cl_pairs;

} UserConstraints;

typedef struct HResult {

	double h_obj;
	double lb_mss;
	double ub_mss;
	double h_time;
	double lb_time;
	double ub_time;
	double all_time;

} HResult;

void save_to_file(arma::mat X, std::string name);
std::map<int, arma::mat> read_part_data(int n, int d, int k, int p, arma::mat Ws);

// compute lb and compute ub, print lb and ub sol
double compute_lb(std::map<int, arma::mat> &sol_map, int k);
double compute_ub(arma::mat Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map, int k, int p);
int generate_part_constraints(std::map<int, arma::mat> sol_map, int k, int p, UserConstraints &constraints);
arma::mat save_lb(std::map<int, arma::mat> &sol_map, int p);
arma::mat save_ub(arma::mat data, arma::mat sol);

HResult heuristic(arma::mat Ws, int p, int k);

#endif //CLUSTERING_AC_HEURISTICS_H