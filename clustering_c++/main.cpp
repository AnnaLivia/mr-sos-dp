#include <iostream>
#include <map>
#include <algorithm>
#include <armadillo>
#include "sdp_branch_and_bound.h"

// data full path
const char *data_path;
const char *sol_path;
const char *constraints_path;
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

// read data Ws
arma::mat read_data(const char *filename, int &n, int &d, int &k) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    // read the header n, d, k
    file >> n >> d >> k;
    arma::mat Ws(n, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            file >> Ws(i, j);
        }
    }
    return Ws;
}

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

// read initial Kmean sol
arma::mat read_sol(const char *filename, int n, int d, int k, double &sol_v) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    // read the header n1, d1, k1, sol_v
    int n1, d1, k1;
    file >> n1 >> d1 >> k1 >> sol_v;
    if (n1 != n || d1 != d || k1 != k) {
        std::cerr << "Error in the initial solution data.\n";
        exit(EXIT_FAILURE);
    }

    arma::mat sol(n, d);
    // read sol (to do)

    return sol;
}

UserConstraints generate_constraints(arma::mat data, double ray){

    double gamma, delta;
    gamma = delta = -1;

    int n = data.n_rows;
    arma::mat distances = arma::zeros(n, n);
    double dist;
    for (int i = 0; i < n; i++){
        arma::vec point_i = data.row(i).t();
        for (int j = i+1; j < n; j++){
            arma::vec point_j = data.row(j).t();
            dist = std::pow(arma::norm(point_i - point_j, 2), 2);
            distances(i, j) = dist;
            distances(j, i) = dist;
        }
    }

    // Normalize the matrix
    double min_val = distances.min();
    double max_val = distances.max();
    distances = (distances - min_val) / (max_val - min_val);

    UserConstraints constraints;
    for (int i = 0; i < n; i++){
        for (int j = i+1; j < n; j++){
            if (distances(i,j) <= ray) {
                std::pair<int,int> ab_pair(i,j);
                constraints.ml_pairs.push_back(ab_pair);
            }
            // else
            //     constraints.cl_pairs.push_back(ab_pair);
        }
    }

    if (delta == -1)
        delta = 0;
    if (gamma == -1)
        gamma = std::numeric_limits<double>::infinity();

    constraints.delta = delta;
    constraints.gamma = gamma;

    return constraints;
}

void save_X_to_file(arma::sp_mat &X){

    std::ofstream f;
    f.open(result_path);
    for (int i = 0; i < X.n_rows; i++){
        int val = X(i,0);
        f << val;
        for (int j = 1; j < X.n_cols; j++){
            val = X(i,j);
            f << " " << val;
        }
        f << "\n";
    }
    f.close();
}

void run(int argc, char **argv) {

    std::string config_file = "config.txt";
    std::map<std::string, std::string> config_map = read_params(config_file);

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
    kmeans_max_iter = 100;
    kmeans_n_start = 200;
    kmeans_verbose = 0;


    if (argc != 6) {
        std::cerr << "Input: <DATA_PATH> <SOL_PATH> <K> <LOG_PATH> <RESULT_PATH>" << std::endl;
        exit(EXIT_FAILURE);
    }

    data_path = argv[1];
    sol_path = argv[2];
    int k = std::stoi(argv[3]);

    int n, d;
    double best_sol_v;
    arma::mat Ws = read_data(data_path, n, d, k);

    string[] km_sol = {"brutto", "meno_brutto", "opt"};
    double all_rays[] = {0.85, 0.75, 0.65};

    double best_ray;
    double ray_sol_v;
    double v_imp;
    arma::sp_mat best_sol;
    arma::sp_mat ray_sol;

    for (string km : km_sol) {

        log_path = argv[4] + km;
        result_path = argv[5] + km;

        log_file.open(log_path);

        log_file << "DATA_PATH, SOL_PATH, n, d, k: ";
        log_file << data_path << " " << sol_path << " " << n << " " << d << " " << k << "\n";
        log_file << "LOG_PATH: " << log_path << "\n\n";

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

        arma::mat sol = read_sol(sol_path, n, d, k, best_sol_v);
        best_ray = -1.0;
        ray_sol_v = std::numeric_limits<double>::infinity();

        // to edit
        sol = Ws;
        for (double ray: all_rays) {
            log_file << "RAY " << ray << ":\n";
            UserConstraints constraints = generate_constraints(sol, ray);
            ray_sol_v = sdp_branch_and_bound(k, Ws, constraints, ray_sol);
            v_imp = (best_sol_v - ray_sol_v) / best_sol_v;
            if (v_imp >= 0) {
                std::cout << "best found";
                best_ray = ray;
                best_sol = ray_sol;
                if (v_imp < 0.01)
                    std::cerr << "Pruning: ray " << ray << ".\n";
                break;
            }
        }

        if (best_ray == -1) {
            std::cerr << "WARNING: Useless instance.\n";
            exit(EXIT_FAILURE);
        }

        save_X_to_file(best_sol);

    }

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
