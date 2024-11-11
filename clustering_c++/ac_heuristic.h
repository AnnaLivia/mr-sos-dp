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
	arma::mat init_sol;
	arma::mat heu_sol;
	int it;

} HResult;

arma::mat save_ub(arma::mat &data, arma::mat &sol);
void save_to_file(arma::mat &X, std::string name);
double compute_ub(arma::mat &Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map);
int generate_part_constraints(std::map<int, arma::mat> &sol_map, UserConstraints &constraints);

void heuristic_kmeans(arma::mat &Ws, HResult &results);
void exact(arma::mat &Ws, HResult &results);
arma::mat create_first_sol(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);
arma::mat evaluate_swap(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &new_sol);
void update(arma::mat &antic_sol, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes);


#endif //CLUSTERING_AC_HEURISTICS_H