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

std::pair<int, std::unordered_map<int, std::vector<int>>> compute_anti_single_cluster(int num_rep, std::vector<int> &cls_points, std::vector<std::vector<double>> &all_dist, int p, int n, double max_d);

#endif //CLUSTERING_AC_HEURISTICS_H