#include <iostream>
#include <filesystem>
#include <map>
#include <list>
#include <algorithm>
#include <armadillo>
#include "Kmeans.h"
#include "kmeans_util.h"
//#include "mr_heuristic.h"
#include "ac_heuristic.h"

// data file
const char *data_path;
const char *opt_path;
const char *sol_path;

// log and result path
std::string result_folder;
std::string result_path;
std::ofstream log_file;
std::ofstream lb_file;
std::ofstream ub_file;

// partition method
char part_m;

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
arma::mat init_sol;

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

double compute_mss(arma::mat data, arma::mat sol) {

    int n = data.n_rows;
    int d = data.n_cols;
    int k = sol.n_cols;

    arma::mat assignment_mat = arma::zeros(n, k);
    arma::vec count = arma::zeros(k);
    arma::mat centroids = arma::zeros(k, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (sol(i,j) == 1) {
                assignment_mat(i, j) = 1;
                ++count(j);
                centroids.row(j) += data.row(i);
            }
        }
    }

    // compute clusters' centroids
    for (int j = 0; j < k; j++) {
        // empty cluster
        if (count(j) == 0) {
            std::printf("read_data(): cluster %d is empty!\n", j);
            return false;
        }
        centroids.row(j) = centroids.row(j) / count(j);
    }

    arma::mat m = data - assignment_mat * centroids;

    return arma::dot(m.as_col(), m.as_col());
}


void run(int argc, char **argv) {
    
    std::string config_file = "config.txt";
    std::map <std::string, std::string> config_map = read_params(config_file);

    result_folder = config_map["RESULT_FOLDER"];

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
    
    if (argc != 8) {
        std::cerr << "Input: <DATA_FILE> <OPT_SOL_FILE> <H_SOL_FILE> <K> <FLIP> <P> <METHOD>" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    data_path = argv[1];
    opt_path = argv[2];
    sol_path = argv[3];
    
    int n, d;
    int k = std::stoi(argv[4]);
    int h = std::stoi(argv[5]);
    int p = std::stoi(argv[6]);
    part_m = *(argv[7]);

    
    std::string str_path = data_path;
    std::string inst_name = str_path.substr(str_path.find_last_of("/\\")+1);
    inst_name = inst_name.substr(0, inst_name.find("."));
    std::ofstream test_SUMMARY(result_folder.substr(0, result_folder.find_last_of("/\\")) + "/test_SUMMARY4.txt", std::ios::app);

    result_path = result_folder + "/" + std::to_string(p) + "part/" + inst_name + "_" + std::to_string(k);
    if (!std::filesystem::exists(result_path))
        std::filesystem::create_directories(result_path);
    result_path += "/" + inst_name + "_" + std::to_string(k) + "_" + part_m;

    if (!std::strchr("crfka", part_m)) {
        std::printf("ERROR: invalid partition method!\n");
        exit(EXIT_FAILURE);
    }

    //lb_file.open(result_path + "_LB.txt");
    //ub_file.open(result_path + "_UB.txt");
    log_file.open(result_path + "_LOG3.txt");

    arma::mat Ws = read_data(data_path, n, d);
    //arma::mat opt_sol = read_sol(opt_path, n, k);
    //double opt_mss = compute_mss(Ws, opt_sol);
    arma::mat opt_sol = arma::mat(n,k);
    double opt_mss = 0;
    if (h == -1)
        init_sol = read_sol(sol_path, n, k);
    else if (h > 0) {
        init_sol = opt_sol;
        flip(init_sol, h);
    } else {
        std::map<int, std::set<int>> ml_map = {};
        std::vector <std::pair<int, int>> local_cl = {};
        std::vector <std::pair<int, int>> global_ml = {};
        std::vector <std::pair<int, int>> global_cl = {};
        Kmeans kmeans(Ws, k, ml_map, local_cl, global_ml, global_cl, kmeans_verbose);
        kmeans.start(kmeans_max_iter, kmeans_n_start, kmeans_permutations);
        std::cout << std::endl << "** Done computing initial Kmean solution **" << std::endl;
        std::cout << "Iter:" << kmeans_max_iter << std::endl << "Start:" << kmeans_n_start;
        std::cout << std::endl << "Permutation:" << kmeans_permutations;
        init_sol = kmeans.getAssignments();
    }
    double init_mss = compute_mss(Ws, init_sol);
    std::cout << std::endl << std::endl;
    std::cout << std::endl << "******************************************************************" << std::endl;
    std::cout << "Instance " << inst_name << std::endl;
    std::cout << "Num Points " << n << std::endl;
    std::cout << "Num Features " << d << std::endl;
    std::cout << "Num Partitions " << p << std::endl;
    std::cout << "Num Clusters " << k << std::endl << std::endl;
    std::cout << "Heuristic MSS: " << init_mss << std::endl;
    std::cout << "Optimal MSS:" << opt_mss << std::endl;
    std::cout << "******************************************************************" << std::endl << std::endl;

    log_file << "DATA_FILE, SOL_FILE, n, d, k: ";
    log_file << data_path << " " << sol_path << " " << n << " " << d << " " << k << "\n";
    
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
    log_file << "optimal MSS: " << opt_mss << "\n";
    log_file << "Heuristic MSS: " << init_mss << "\n\n";

    HResult results;
    test_SUMMARY << inst_name << "\t"
    << k << "\t"
    << p << "\t"
    << opt_mss << "\t";

    part_m = 'o';
    log_file << "Method cluster-part model \n" << "\n";
    results = heuristic(Ws, p, k);

    test_SUMMARY << part_m << "\t"
    << results.h_obj << "\t"
    << results.lb_mss << "\t"
    << round((init_mss - results.lb_mss) / init_mss * 100) << "\t"
    << round(results.h_time) << "\t"
    << round(results.lb_time) << "\t"
    << round(results.ub_time) << "\t"
    << round(results.all_time) << "\n";

    std::cout << std::endl << "--------------------------------------------------------------------";
    std::cout << std::endl << "Method " << part_m << " GAP SOL-LB " <<  round((init_mss - results.lb_mss) / init_mss * 100) << "%" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    /*
    part_m = 'a';
    log_file << "Method antic total \n" << "\n";
    bb = test_lb(Ws, p, k);

    test_SUMMARY << part_m << "\t"
    << bb.first << "\t"
    << round((init_mss - bb.first) / init_mss * 100) << "\t"
    << round((bb.second - bb.first) / bb.second * 100) << "\n";

    std::cout << std::endl << "--------------------------------------------------------------------";
    std::cout << std::endl << "Method " << part_m << " GAP SOL-LB " <<  round((init_mss - bb.first) / init_mss * 100) << "%" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;


    part_m = 'k';
    log_file << "Method a \n" << test_lb(Ws, p, k) << "\n";
    part_m = 'c';
    log_file << "Method c \n" << test_lb(Ws, p, k) << "\n";
    part_m = 'r';
    log_file << "Method r \n" << test_lb(Ws, p, k) << "\n";
    part_m = 'f';
    << sol_path << "\t"

    ResultData results = mr_heuristic(k, p, Ws);

    int it = results.it;
    double ub_mss = results.ub_mss;
    double lb_mss = results.lb_mss;
    double ub_time = results.ub_time;
    int lb_update = results.lb_update;
    int ub_update = results.ub_update;
    int ray_lb_update = results.ray_lb_update;
    int ray_ub_update = results.ray_ub_update;
    double lb_time = results.lb_time;
    double ray_time = results.ray_time;
    double all_time = results.all_time;

    std::cout << std::endl << "**********************************************************" << std::endl;
    std::cout << "Optimal MSS BOUND " << opt_mss << std::endl;
    std::cout << "Heuristic MSS BOUND " << init_mss << std::endl;
    std::cout << "Best LB MSS BOUND " << lb_mss << std::endl;
    std::cout << "Best UB MSS BOUND " << ub_mss << std::endl;
    std::cout << "GAP UB-LB " << round((ub_mss - lb_mss) / ub_mss * 100) << "%" << std::endl;
    std::cout << "Num It " << it << std::endl;
    std::cout << "Num LB update " << lb_update << std::endl;
    std::cout << "Num UB update " << ub_update << std::endl;
    std::cout << "Num RAY LB update " << ray_lb_update << std::endl;
    std::cout << "Num RAY UB update " << ray_ub_update << std::endl;
    std::cout << "LB Time " << lb_time << " sec" << std::endl;
    std::cout << "UB Time " << ub_time << " sec" << std::endl;
    std::cout << "RAY Time " << ray_time << " sec" << std::endl;
    std::cout << "ALL Time " << all_time << " sec" << std::endl;
    std::cout << "GAP LB Opt " << round((opt_mss - lb_mss) / opt_mss * 100) << "%" << std::endl;
    std::cout << "GAP UB Opt " << round((ub_mss - opt_mss) / opt_mss * 100) << "%" << std::endl;
    std::cout << "GAP LB Heur " << round((init_mss - lb_mss) / init_mss * 100) << "%" << std::endl;
    std::cout << "GAP UB Heur " << round((ub_mss - init_mss) / init_mss * 100) << "%" << std::endl;
    std::cout << "**********************************************************" << std::endl << std::endl;

	test_SUMMARY << inst_name << "\t"
	<< p << "\t"
	<< k << "\t"
    << opt_mss << "\t"
    << init_mss << "\t"
    << lb_mss << "\t"
    << ub_mss << "\t"
    << round((ub_mss - lb_mss) / ub_mss * 100) << "%" << "\t"
    << it << "\t"
    << lb_update << "\t"
    << ub_update << "\t"
    << ray_lb_update << "\t"
    << ray_ub_update << "\t"
    << lb_time << "\t"
    << ub_time << "\t"
    << ray_time << "\t"
    << all_time << "\t"
    << round((opt_mss - lb_mss) / opt_mss * 100) << "%" << "\t"
    << round((ub_mss - opt_mss) / opt_mss * 100) << "%" << "\t"
    << round((init_mss - lb_mss) / init_mss * 100) << "%" << "\t"
    << round((ub_mss - init_mss) / init_mss * 100) << "%" << "\t"
    << "\n";

    */

    log_file.close();
	test_SUMMARY.close();

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
