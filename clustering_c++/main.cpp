#include <iostream>
#include <map>
#include <list>
#include <algorithm>
#include <armadillo>
#include "Kmeans.h"
#include "kmeans_util.h"
#include "mr_heuristic.h"

// data file and path
const char *data_path;
const char *opt_path;
const char *sol_path;
const char *log_path;
const char *result_path;
std::ofstream log_file;

// branch and bound
double branch_and_bound_tol;
int branch_and_bound_parallel;
int branch_and_bound_max_nodes;
int branch_and_bound_visiting_strategy;

// sdp solver
int sdp_solver_session_threads_root;
int sdp_solver_session_threads;
const char *sdp_solver_folder;
double sdp_solver_tol;
int sdp_solver_stopoption;
int sdp_solver_maxiter;
int sdp_solver_maxtime;
int sdp_solver_verbose;
int sdp_solver_max_cp_iter_root;
int sdp_solver_max_cp_iter;
double sdp_solver_cp_tol;
int sdp_solver_max_ineq;
double sdp_solver_inherit_perc;
double sdp_solver_eps_ineq;
double sdp_solver_eps_active;
int sdp_solver_max_pair_ineq;
double sdp_solver_pair_perc;
int sdp_solver_max_triangle_ineq;
double sdp_solver_triangle_perc;

// heuristic
bool kmeans_sdp_based;
int kmeans_max_iter;
int kmeans_n_start;
int kmeans_permutations;
bool kmeans_verbose;

// read parameters in config file
std::map<std::string, std::string> read_params(std::string &config_file) {

    std::map<std::string, std::string> config_map = {};

    std::ifstream cFile (config_file);
    if (cFile.is_open()) {
        std::string line;
        while (getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find('=');
            auto key = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            config_map.insert(std::pair<std::string, std::string>(key, value));
        }

    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }

    return config_map;
}

// read data Ws
arma::mat read_data(const char *filename, int &n, int &d) {

    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    // read the header n, d
    file >> n >> d;
    arma::mat data(n, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            file >> data(i, j);
        }
    }

    return data;
}

// read initial sol
arma::mat read_sol(const char *filename, int n, int k) {

    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    arma::mat sol(n, k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            file >> sol(i, j);
        }
    }
    
    return sol;
}

// flip points
void flip(arma::mat &sol, int f) {

    int n = sol.n_rows;
    int k = sol.n_cols;
    int cls;
    int rand_c;
    int i;
    
    for (int l = 0; l < f; l++) {
        i = std::rand() % (n);
        for (int j = 0; j < k; j++) {
            if (sol(i, j) == 1.0) {
                cls = j;
                break;
            }
        }
        sol.row(i) = arma::zeros(k).t();
        do {
            rand_c = rand() % (k); // Generate a new random cluster
        } while (rand_c == cls);
        sol(i, rand_c) = 1.0;
    }
    
    std::cout << std::endl << "** Done flipping " << f << " points **" << std::endl;
}

void run(int argc, char **argv) {
    
    std::string config_file = "config.txt";
    std::map <std::string, std::string> config_map = read_params(config_file);
    
    // branch and bound
    branch_and_bound_tol = std::stod(config_map["BRANCH_AND_BOUND_TOL"]);
    branch_and_bound_parallel = std::stoi(config_map["BRANCH_AND_BOUND_PARALLEL"]);
    branch_and_bound_max_nodes = std::stoi(config_map["BRANCH_AND_BOUND_MAX_NODES"]);
    branch_and_bound_visiting_strategy = std::stoi(config_map["BRANCH_AND_BOUND_VISITING_STRATEGY"]);
    
    // sdp solver
    // sdp_solver_matlab_session = config_map["SDP_SOLVER_MATLAB_SESSION"].c_str();
    sdp_solver_session_threads_root = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS_ROOT"]);
    sdp_solver_session_threads = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS"]);
    sdp_solver_folder = config_map["SDP_SOLVER_FOLDER"].c_str();
    sdp_solver_tol = std::stod(config_map["SDP_SOLVER_TOL"]);
    sdp_solver_verbose = std::stoi(config_map["SDP_SOLVER_VERBOSE"]);
    sdp_solver_max_cp_iter_root = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER_ROOT"]);
    sdp_solver_max_cp_iter = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER"]);
    sdp_solver_cp_tol = std::stod(config_map["SDP_SOLVER_CP_TOL"]);
    sdp_solver_max_ineq = std::stoi(config_map["SDP_SOLVER_MAX_INEQ"]);
    sdp_solver_inherit_perc = std::stod(config_map["SDP_SOLVER_INHERIT_PERC"]);
    sdp_solver_eps_ineq = std::stod(config_map["SDP_SOLVER_EPS_INEQ"]);
    sdp_solver_eps_active = std::stod(config_map["SDP_SOLVER_EPS_ACTIVE"]);
    sdp_solver_max_pair_ineq = std::stoi(config_map["SDP_SOLVER_MAX_PAIR_INEQ"]);
    sdp_solver_pair_perc = std::stod(config_map["SDP_SOLVER_PAIR_PERC"]);
    sdp_solver_max_triangle_ineq = std::stoi(config_map["SDP_SOLVER_MAX_TRIANGLE_INEQ"]);
    sdp_solver_triangle_perc = std::stod(config_map["SDP_SOLVER_TRIANGLE_PERC"]);
    sdp_solver_stopoption = 0;
    sdp_solver_maxiter = 50000;
    sdp_solver_maxtime = 3600;
    
    // kmeans
    kmeans_max_iter = 200;
    kmeans_n_start = 100;
    kmeans_verbose = 0;
    kmeans_permutations = 1;
    
    if (argc != 9) {
        std::cerr << "Input: <DATA_FILE> <OPT_SOL_FILE> <H_SOL_FILE> <LOG_FILE> <RESULT_PATH> <K> <F> <P>" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    data_path = argv[1];
    opt_path = argv[2];
    sol_path = argv[3];
    log_path = argv[4];
    result_path = argv[5];
    
    int n, d;
    int k = std::stoi(argv[6]);
    int f = std::stoi(argv[7]);
    int p = std::stoi(argv[8]);
    
    arma::mat Ws = read_data(data_path, n, d);
    arma::mat opt_sol = read_sol(opt_path, n, k);
    double opt_mss = compute_mss(Ws, opt_sol);
    arma::mat init_sol;
    if (f == -1) {
        init_sol = read_sol(sol_path, n, k);
    }
    else if (f > 0) {
        init_sol = opt_sol;
        flip(init_sol, f);
    } else {
        std::map<int, std::set<int>> ml_map = {};
        std::vector <std::pair<int, int>> local_cl = {};
        std::vector <std::pair<int, int>> global_ml = {};
        std::vector <std::pair<int, int>> global_cl = {};
        Kmeans kmeans(Ws, k, ml_map, local_cl, global_ml, global_cl, kmeans_verbose);
        kmeans.start(kmeans_max_iter, kmeans_n_start, kmeans_permutations);
        std::cout << std::endl << "** Done computing initial Kmean solution **" << std::endl;
        std::cout << "Iter:" << kmeans_max_iter << std::endl << "Start:" << kmeans_n_start;
        std::cout << std::endl << "Permutation:" << kmeans_permutations << std::endl << std::endl;
        init_sol = kmeans.getAssignments();
    }
    save_to_file(init_sol, data_path, "");
    std::map < int, std::list < std::pair < int, double>>> cls_map;
    double init_mss = compute_clusters(Ws, init_sol, cls_map);
    std::cout << std::endl << "**********************************************************" << std::endl;
    std::cout << "Heuristic MSS " << init_mss << std::endl;
    std::cout << "Optimal MSS:" << opt_mss << std::endl << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "# Partitions = " << p << std::endl << std::endl;
    
    log_file.open(log_path);
    log_file << "DATA_FILE, SOL_FILE, n, d, k: ";
    log_file << data_path << " " << sol_path << " " << n << " " << d << " " << k << "\n";
    log_file << "LOG_FILE: " << log_path << "\n\n";
    
    log_file << "BRANCH_AND_BOUND_TOL: " << branch_and_bound_tol << "\n";
    log_file << "BRANCH_AND_BOUND_PARALLEL: " << branch_and_bound_parallel << "\n";
    log_file << "BRANCH_AND_BOUND_MAX_NODES: " << branch_and_bound_max_nodes << "\n";
    log_file << "BRANCH_AND_BOUND_VISITING_STRATEGY: " << branch_and_bound_visiting_strategy << "\n\n";
    
    log_file << "SDP_SOLVER_SESSION_THREADS_ROOT: " << sdp_solver_session_threads_root << "\n";
    log_file << "SDP_SOLVER_SESSION_THREADS: " << sdp_solver_session_threads << "\n";
    log_file << "SDP_SOLVER_FOLDER: " << sdp_solver_folder << "\n";
    log_file << "SDP_SOLVER_TOL: " << sdp_solver_tol << "\n";
    log_file << "SDP_SOLVER_VERBOSE: " << sdp_solver_verbose << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER_ROOT: " << sdp_solver_max_cp_iter_root << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER: " << sdp_solver_max_cp_iter << "\n";
    log_file << "SDP_SOLVER_CP_TOL: " << sdp_solver_cp_tol << "\n";
    log_file << "SDP_SOLVER_MAX_INEQ: " << sdp_solver_max_ineq << "\n";
    log_file << "SDP_SOLVER_INHERIT_PERC: " << sdp_solver_inherit_perc << "\n";
    log_file << "SDP_SOLVER_EPS_INEQ: " << sdp_solver_eps_ineq << "\n";
    log_file << "SDP_SOLVER_EPS_ACTIVE: " << sdp_solver_eps_active << "\n";
    log_file << "SDP_SOLVER_MAX_PAIR_INEQ: " << sdp_solver_max_pair_ineq << "\n";
    log_file << "SDP_SOLVER_PAIR_PERC: " << sdp_solver_pair_perc << "\n";
    log_file << "SDP_SOLVER_MAX_TRIANGLE_INEQ: " << sdp_solver_max_triangle_ineq << "\n";
    log_file << "SDP_SOLVER_TRIANGLE_PERC: " << sdp_solver_triangle_perc << "\n\n";
    log_file << "Heuristic MSS: " << init_mss << "\n\n";
    
    std::pair<double,double> bounds = mr_heuristic(k, p, Ws, data_path);
    double lb_mss = bounds.first;
    double ub_mss = bounds.second;

    std::cout << std::endl << "**********************************************************" << std::endl;
    std::cout << "Optimal MSS BOUND " << opt_mss << std::endl;
    std::cout << "Heuristic MSS BOUND " << init_mss << std::endl;
    std::cout << "Best LB MSS BOUND " << lb_mss << std::endl;
    std::cout << "Best UB MSS BOUND " << ub_mss << std::endl;
    std::cout << "GAP LB Opt " << round((opt_mss - lb_mss) / opt_mss * 100) << "%" << std::endl;
    std::cout << "GAP UB Opt " << round((ub_mss - opt_mss) / opt_mss * 100) << "%" << std::endl;
    std::cout << "GAP LB Heur " << round((init_mss - lb_mss) / init_mss * 100) << "%" << std::endl;
    std::cout << "GAP UB Heur " << round((ub_mss - init_mss) / init_mss * 100) << "%" << std::endl;
    std::cout << std::endl << "**********************************************************" << std::endl;

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
