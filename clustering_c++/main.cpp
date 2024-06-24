#include <iostream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <armadillo>
#include "Kmeans.h"
#include "kmeans_util.h"
#include "ac_heuristic.h"

// data file
const char *data_path;
const char *opt_path;

// log and result path
std::string result_folder;
std::string result_path;
std::ofstream log_file;
std::ofstream lb_file;
std::ofstream ub_file;

// instance data
int n;
int d;
int p;
int k;
bool stddata;

// partition and anticlustering
double w_diversity;
double w_dispersion;
int num_rep;
int n_threads_anti;
int n_threads_part;

// heuristic
int kmeans_max_it;
int kmeans_start;
int kmeans_permut;
bool kmeans_verbose;
arma::mat init_sol;

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
arma::mat read_data(const char *filename) {

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

    if (!stddata) {
        arma::mat dist = compute_distances(data);
        double max_d = arma::max(arma::max(dist));
        if (max_d != 0) {
            int order = std::floor(std::log10(max_d))/2;
            std::cout << "Scaling order > 3 " << "\n\n\n";
            if (order >= 2)
                for (int j = 0; j < d; j++)
                    data.col(j) *= std::pow(10, - order);
        }
    }
    else {
    	// normalized data between 0 and 1
        std::cout << "Scaling in 0-1 each feature" << "\n\n\n";
        for (int j = 0; j < d; j++) {
        	double mean = arma::mean(data.col(j));
        	double stddev = arma::stddev(data.col(j));
        	if (mean != 0 and stddev!=1) {
        		if (stddev != 0)
            		data.col(j) = (data.col(j) - mean) / stddev;
        		else
            		data.col(j).zeros();
            }
        }
    }

    return data;
}

// read initial sol
arma::mat read_sol(const char *filename) {

    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }
    
    std::ifstream filecheck(filename);
    std::string line;
    getline(filecheck, line);
    std::stringstream ss(line);
    int cols = 0;
    double item;
    while(ss >> item) cols++;
    filecheck.close();

    arma::mat sol(n, k);
    for (int i = 0; i < n; i++) {
        if (cols == 1) {
            sol.row(i) = arma::zeros(k).t();
            int c = 0;
            file >> c;
            sol(i, c) = 1;
        }
        else {
            for (int j = 0; j < k; j++)
                file >> sol(i, j);
        }
    }
    
    return sol;
}

/*
std::vector<int> read_assignment(const char *filename) {
    std::ifstream file(filename);
    if (!file) {
        //std::cerr << strerror(errno) << "\n";
        //exit(EXIT_FAILURE);
        return {};
    }
    std::vector<int> v(n);
    for (int i = 0; i < n; i++) {
        file >> v[i];
    }
    return v;
}
*/

void run(int argc, char **argv) {
    
    std::string config_file = "config.txt";
    std::map <std::string, std::string> config_map = read_params(config_file);

    result_folder = config_map["RESULT_FOLDER"];

    // number of thread for computing the partition bound
    w_diversity = std::stoi(config_map["ANTICLUSTERING_DIVERSITY"]);
    w_dispersion = std::stoi(config_map["ANTICLUSTERING_DISPERSION"]);
    num_rep = std::stoi(config_map["ANTICLUSTERING_REP"]);
    n_threads_anti = std::stoi(config_map["ANTICLUSTERING_THREADS"]);
    n_threads_part = std::stoi(config_map["PARTITION_THREADS"]);

    // branch and bound
    branch_and_bound_tol = std::stod(config_map["BRANCH_AND_BOUND_TOL"]);
    branch_and_bound_parallel = std::stoi(config_map["BRANCH_AND_BOUND_PARALLEL"]);
    branch_and_bound_max_nodes = std::stoi(config_map["BRANCH_AND_BOUND_MAX_NODES"]);
    branch_and_bound_visiting_strategy = std::stoi(config_map["BRANCH_AND_BOUND_VISITING_STRATEGY"]);

    // sdp solver
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
    kmeans_start = std::stoi(config_map["KMEANS_NUM_START"]);
    kmeans_max_it = std::stoi(config_map["KMEANS_MAX_ITERATION"]);
    kmeans_verbose = std::stoi(config_map["KMEANS_VERBOSE"]);
    kmeans_permut = 1;
    
    if (argc != 6) {
        std::cerr << "Input: <DATA_FILE> <OPT_SOL_FILE> <K> <P> <STD>" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    data_path = argv[1];
    opt_path = argv[2];

    k = std::stoi(argv[3]);
    p = std::stoi(argv[4]);
    stddata = std::stoi(argv[5]);

    std::string str_path = data_path;
    std::string inst_name = str_path.substr(str_path.find_last_of("/\\")+1);
    inst_name = inst_name.substr(0, inst_name.find("."));
    std::ofstream test_SUMMARY(result_folder.substr(0, result_folder.find_last_of("/\\")) + "/test_SUMMARY.txt", std::ios::app);

    result_path = result_folder + "/" + std::to_string(p) + "part/" + inst_name + "_" + std::to_string(k);
    if (!std::filesystem::exists(result_path))
        std::filesystem::create_directories(result_path);
    result_path += "/" + inst_name + "_" + std::to_string(k);

    log_file.open(result_path + "_LOG.txt");

    arma::mat Ws = read_data(data_path);
    //arma::mat opt_sol = read_sol(opt_path);
    //double opt_mss = compute_mss(Ws, opt_sol);
    arma::mat opt_sol = arma::mat(n, k);
    double opt_mss;

    std::map<int, std::set<int>> ml_map = {};
    std::vector <std::pair<int, int>> local_cl = {};
    std::vector <std::pair<int, int>> global_ml = {};
    std::vector <std::pair<int, int>> global_cl = {};
    Kmeans kmeans(Ws, k, ml_map, local_cl, global_ml, global_cl, kmeans_verbose);
    kmeans.start(kmeans_max_it, kmeans_start, kmeans_permut);
    init_sol = kmeans.getAssignments();
    double init_mss = compute_mss(Ws, init_sol);


    std::cout << std::endl << "---------------------------------------------------------------------" << std::endl;
    std::cout << "Instance " << inst_name << std::endl;
    std::cout << "Num Points " << n << std::endl;
    std::cout << "Num Features " << d << std::endl;
    std::cout << "Num Partitions " << p << std::endl;
    std::cout << "Num Clusters " << k << std::endl << std::endl;
    std::cout << "Optimal MSS:" << std::fixed << std::setprecision(1) << opt_mss << std::endl;
    std::cout << "Heuristic MSS: " << init_mss << "  (num_starts " << kmeans_start << ")" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl << std::endl;

    log_file << "DATA_FILE, n, d, k: ";
    log_file << data_path << " " << n << " " << d << " " << k << "\n";

    log_file << "ANTICLUSTERING_REP: " << num_rep << "\n\n";
    log_file << "ANTICLUSTERING_THREADS: " << n_threads_anti << "\n\n";
    log_file << "PARTITION_THREADS: " << n_threads_part << "\n";

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
    log_file << std::fixed <<  std::setprecision(1);
    log_file << "Optimal MSS: " << opt_mss << "\n";
    log_file << "Heuristic MSS: " << init_mss << "\n\n";

    HResult results = heuristic(Ws);
    test_SUMMARY << std::fixed <<  std::setprecision(1)
    << inst_name << "\t"
    << k << "\t"
    << p << "\t"
    << results.h_obj << "\t"
    << opt_mss << "\t"
    << init_mss << "\t"
    << results.lb_mss << "\t"
    << (init_mss - results.lb_mss) / init_mss * 100 << "%\t"
    << results.ub_mss << "\t"
    << (init_mss - results.ub_mss) / init_mss * 100 << "%\t"
    << results.h_time << "\t"
    << results.lb_time << "\t"
    << results.h_time/60 + results.lb_time/60 << "\n";

    std::cout << std::endl << "---------------------------------------------------------------------";
    std::cout << std::endl << "OPT: " << opt_mss << std::endl;
    std::cout << "HEU: " << init_mss << std::endl;
    std::cout << "LB MSS: " << results.lb_mss << std::endl;
    std::cout << std::endl << "HEU - LB " << init_mss - results.lb_mss << std::endl;
    std::cout << "GAP HEU-LB " <<  (init_mss - results.lb_mss) / init_mss * 100 << "%" << std::endl;
    std::cout << std::endl << "HEU - UB " << init_mss - results.ub_mss << std::endl;
    std::cout << "GAP HEU-UB " <<  (init_mss - results.ub_mss) / init_mss * 100 << "%" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;

    log_file.close();
	test_SUMMARY.close();

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
