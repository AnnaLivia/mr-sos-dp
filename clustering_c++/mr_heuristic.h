//
// Created by moden on 08/04/2024.
//
#ifndef CLUSTERING_MR_HEURISTICS_H
#define CLUSTERING_MR_HEURISTICS_H

#include <armadillo>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"

typedef struct UserConstraints {

	std::vector<std::pair<int,int>> ml_pairs, cl_pairs;

} UserConstraints;

typedef struct ResultData {

	int it;
	double lb_mss;
	double ub_mss;
	int ub_update;
	int lb_update;
	int ray_ub_update;
	int ray_lb_update;
	double lb_time;
	double ub_time;
	double ray_time;
	double all_time;

} ResultData;

void save_to_file(arma::mat X, std::string name);
std::map<int, arma::mat> read_part_data(int n, int d, int k, int p);

double compute_clusters(arma::mat data, arma::mat sol, std::map<int, std::list<std::pair<int, double>>> &cls_map);

// generate must link constraints
UserConstraints generate_constraints(std::map<int, std::list<std::pair<int, double>>> cls_map, double ray);

// generate must link constraints on partition sol
int generate_part_constraints(std::map<int, arma::mat> sol_map, int k, int p, UserConstraints &constraints);
double compute_part_lb(std::map<int, arma::mat> &sol_map);
std::map<int, arma::mat> compute_comb_bound(arma::mat &data, int p);

// generate partitions from clusters
std::map<int, arma::mat> generate_partitions(arma::mat data, int n_part,
                                             std::map<int, std::list<std::pair<int, double>>> cls_map);

// generate partitions taking closest points
void take_n_from_p(arma::mat data, std::map<int, int> &new_p, std::map<int, int> &prec_p, int n);

// generate partitions
std::map<int, arma::mat> generate_partitions(arma::mat data, int p);

// compute lb and compute ub, print lb and ub sol
double compute_lb(std::map<int, arma::mat> &sol_map, int k, int p);
double compute_ub(arma::mat Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map, int k, int p);
arma::mat save_lb(std::map<int, arma::mat> &sol_map, int p);
arma::mat save_ub(arma::mat data, arma::mat sol);

// solve with different rays
double solve_with_ray(arma::mat Ws, arma::mat init_sol, int k, int p, int &ub_update, int &lb_update, double &lb);

// define moving ray routine
ResultData mr_heuristic(int k, int p, arma::mat Ws);

// define moving ray routine only with ray mechanism
ResultData mr_heuristic_only_ray(int k, int p, arma::mat Ws);

#endif //CLUSTERING_MR_HEURISTICS_H