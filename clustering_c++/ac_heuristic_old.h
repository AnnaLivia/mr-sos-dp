#ifndef CLUSTERING_AC_HEURISTICS_H
#define CLUSTERING_AC_HEURISTICS_H

#include <unordered_map>
#include "MatlabEngine.hpp"
#include "MatlabDataArray.hpp"
#include "sdp_branch_and_bound.h"

typedef struct HResult {

	double anti_obj;
	double heu_mss;
	double lb_mss;
	double ub_mss;
	double h_time;
	double m_time;
	double lb_time;
	arma::mat heu_sol;
	int it;

} HResult;

void save_to_file(arma::mat &X, std::string name);
double compute_mss(arma::mat &data, arma::mat &sol);

// compute lb and compute ub, print lb and ub sol
//double compute_lb(std::map<int, arma::mat> &sol_map);
double compute_ub(arma::mat &Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map);
int generate_part_constraints(std::map<int, arma::mat> &sol_map, UserConstraints &constraints);
arma::mat save_lb(std::map<int, arma::mat> &sol_map);
arma::mat save_ub(arma::mat &data, arma::mat &sol);
arma::mat evaluate_anti(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &ub_sol);
void update_sol(arma::mat &antic_sol, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);
arma::mat retrieve_sol(arma::mat &antic_sol, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);
arma::mat create_first_sol(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);
arma::mat retrieve_sol(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, std::vector<arma::mat> &cls_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);
void calculate_sol(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, std::vector<arma::mat> &cls_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);

std::pair<double,  std::vector<std::vector<int>>> compute_anti_cls(std::vector<int> &cls_points, std::vector<std::vector<double>> &all_dist);
void heuristic(arma::mat &Ws, HResult &results);
void heuristic_no_sol(arma::mat &Ws, HResult &results);
void heuristic_kmeans(arma::mat &Ws, HResult &results);
void heuristic_new(arma::mat &Ws, HResult &results);

#endif //CLUSTERING_AC_HEURISTICS_H