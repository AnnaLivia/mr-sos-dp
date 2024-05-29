#ifndef CLUSTERING_AC_HEURISTICS_H
#define CLUSTERING_AC_HEURISTICS_H

#include <unordered_map>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#include "sdp_branch_and_bound.h"

typedef struct HResult {

	double h_obj;
	double lb_mss;
	double ub_mss;
	double h_time;
	double lb_time;

} HResult;

void save_to_file(arma::mat &X, std::string name);
std::map<int, arma::mat> read_part_data(arma::mat &Ws);

// compute lb and compute ub, print lb and ub sol
double compute_lb(std::map<int, arma::mat> &sol_map);
double compute_ub(arma::mat &Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map);
int generate_part_constraints(std::map<int, arma::mat> &sol_map, UserConstraints &constraints);
arma::mat save_lb(std::map<int, arma::mat> &sol_map);
arma::mat save_ub(arma::mat &data, arma::mat &sol);

std::pair<double, std::unordered_map<int, std::vector<int>>> compute_anti_single_cluster(std::vector<int> &cls_points, double max_d, std::vector<std::vector<double>> &all_dist);
HResult heuristic(arma::mat &Ws);

#endif //CLUSTERING_AC_HEURISTICS_H