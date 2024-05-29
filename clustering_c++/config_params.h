#ifndef CLUSTERING_CONFIG_PARAMS_H
#define CLUSTERING_CONFIG_PARAMS_H

#define ROOT -1
#define CANNOT_LINK 0
#define MUST_LINK 1

#define DEFAULT 0
#define ABSOLUTE 1
#define NORM 2

#define BEST_FIRST 0
#define DEPTH_FIRST 1
#define BREADTH_FIRST 2

// cp_flag values
#define CP_FLAG_WORST -3
#define CP_FLAG_NO_SUCCESS -2
#define CP_FLAG_MAX_ITER -1
#define CP_FLAG_NO_VIOL 0
#define CP_FLAG_MAX_INEQ 1
#define CP_FLAG_PRUNING 2
#define CP_FLAG_CP_TOL 3
#define CP_FLAG_INFEAS 4
#define CP_FLAG_SDP_INFEAS 5

// data full path
extern const char *data_path;
extern const char *sol_path;
extern const char *constraints_path;

// log and result path
extern std::string result_folder;
extern std::string result_path;
extern std::ofstream log_file;
extern std::ofstream lb_file;
extern std::ofstream ub_file;

// instance data
extern int n;
extern int d;
extern int k;
extern int p;

// partition and anticlustering
extern int n_threads_part;
extern int n_threads_anti;
extern int num_rep;

// branch and bound
extern double branch_and_bound_tol;
extern int branch_and_bound_parallel;
extern int branch_and_bound_max_nodes;
extern int branch_and_bound_visiting_strategy;

// sdp solver
extern int sdp_solver_session_threads_root;
extern int sdp_solver_session_threads;
extern const char *sdp_solver_folder;
extern double sdp_solver_tol;
extern int sdp_solver_stopoption;
extern int sdp_solver_maxiter;
extern int sdp_solver_maxtime;
extern int sdp_solver_verbose;
extern int sdp_solver_max_cp_iter_root;
extern int sdp_solver_max_cp_iter;
extern double sdp_solver_cp_tol;
extern double sdp_solver_eps_ineq;
extern double sdp_solver_eps_active;
extern int sdp_solver_max_ineq;
extern int sdp_solver_max_pair_ineq;
extern double sdp_solver_pair_perc;
extern int sdp_solver_max_triangle_ineq;
extern double sdp_solver_triangle_perc;
extern double sdp_solver_inherit_perc;

// kmeans
extern int kmeans_max_it;
extern int kmeans_start;
extern bool kmeans_verbose;
extern int kmeans_permut;
extern arma::mat init_sol;

#endif //CLUSTERING_CONFIG_PARAMS_H
