#include <iostream>
#include <map>
#include <list>
#include <algorithm>
#include <armadillo>
#include "sdp_branch_and_bound.h"
#include "Kmeans.h"
#include "kmeans_util.h"

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

double compute_mss(arma::mat &data, arma::mat sol) {
    
    int n = data.n_rows;
    int d = data.n_cols;
    int k = sol.n_cols;

    arma::vec assignments = arma::zeros(n) - 1;
    arma::vec count = arma::zeros(k);
    arma::mat centroids = arma::zeros(k, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (sol(i,j) == 1.0) {
                assignments(i) = j;
                count(j) = count(j) + 1;
                centroids.row(j) += data.row(i);
            }
        }
    }

    // compute clusters' centroids
    for (int j = 0; j < k; ++j) {
        // empty cluster
        if (count(j) == 0) {
            std::printf("read_data(): cluster %d is empty!\n", j);
            return false;
        }
        centroids.row(j) = centroids.row(j) / count(j);
    }
    
    double sol_mss = 0;
    int cluster;
    arma::vec centroid;
    for (int i = 0; i < n; i++) {
        cluster = assignments(i);
        arma::vec point = data.row(i).t();
        centroid = centroids.row(cluster).t();
        sol_mss += squared_distance(point, centroid);
    }
    
    std::cout << "OPT SOL:" << sol_mss << std::endl;
    return sol_mss;
}

double compute_clusters(arma::mat &data, arma::mat sol, std::map<int, std::list<std::pair<int, double>>> &cls_map) {

    int n = data.n_rows;
    int d = data.n_cols;
    int k = sol.n_cols;

    arma::vec assignments = arma::zeros(n) - 1;
    arma::vec count = arma::zeros(k);
    arma::mat centroids = arma::zeros(k, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (sol(i,j) == 1.0) {
                assignments(i) = j;
                count(j) = count(j) + 1;
                centroids.row(j) += data.row(i);
            }
        }
    }

    // compute clusters' centroids
    for (int j = 0; j < k; ++j) {
        // empty cluster
        if (count(j) == 0) {
            std::printf("read_data(): cluster %d is empty!\n", j);
            return false;
        }
        centroids.row(j) = centroids.row(j) / count(j);
        cls_map[j] = {};
    }
    
    int cluster;
    double sol_mss = 0;
    double dist;
    arma::vec maxDist = arma::zeros(k);
    arma::vec minDist = arma::zeros(k);
    arma::vec centroid;
    for (int i = 0; i < n; i++) {
        cluster = assignments(i);
        arma::vec point = data.row(i).t();
        centroid = centroids.row(cluster).t();
        dist = squared_distance(point, centroid);
        if (dist > maxDist(cluster))
            maxDist(cluster) = dist;
        if (dist < minDist(cluster))
            minDist(cluster) = dist;
        sol_mss += dist;
        cls_map[cluster].insert(cls_map[cluster].begin(), std::pair<int, double>(i, dist));
    }

    // Normalize the matrix
    auto compareDist = [](std::pair<int, double>& a, std::pair<int, double>& b) {
        return a.second < b.second;
    };
    std::list<std::pair<int, double>> points;
    for (int j = 0; j < k; j++) {
        points = cls_map[j];
        for (auto &pair : points)
            pair.second = (pair.second - minDist(j)) / (maxDist(j) - minDist(j));
        points.sort(compareDist);
        cls_map[j] = points;
    }

    return sol_mss;
}

// generate must link constraints
UserConstraints generate_constraints(std::map<int, std::list<std::pair<int, double>>> cls_map, double ray) {

    UserConstraints constraints;
    
    int c = 0;
    std::pair<int, double> p;
    std::pair<int, double> q;
    std::list<std::pair<int, double>> points;
    for (auto& cls : cls_map) {
        points = cls.second;
        for (auto it1 = points.begin(); it1 != std::prev(points.end()); ++it1) {
            p = *it1;
            if (p.second <= ray) {
                for (auto it2 = points.begin(); it2 != it1; ++it2) {
                    q = *it2;
                    std::pair<int,int> ab_pair(p.first,q.first);
                    constraints.ml_pairs.push_back(ab_pair);
                    c++;
                }
            }
            else
                break;
        }
    }
    std::cout << "Added constraits: " << c << std::endl << std::endl;

    return constraints;
}

// generate partitions
std::map<int, arma::mat> generate_partitions(arma::mat data, std::map<int, std::list<std::pair<int, double>>> cls_map,
                                             int n_part) {
    
    int n = data.n_rows;
    int d = data.n_cols;
    int k = cls_map.size();
    
    std::map<int, arma::mat> part_map;
    arma::vec n_points = arma::zeros(k);
    for (int i=0; i < n_part; ++i)
        part_map[i] = arma::zeros(n, d);
    
    int part = 0;
    int part_points = 0;
    for (auto& cls : cls_map) {
        for (auto& p : cls.second) {
            part = rand() % (n_part);
            part_points = n_points(part);
            part_map[part].row(part_points) = data.row(p.first);
            n_points(part) += 1;
        }
    }

    for (auto& part : part_map)
        part.second = part.second.submat(0, 0, n_points(part.first) - 1, d - 1);
    
    return part_map;
}


void take_n_from_p(arma::mat data, std::map<int, int> &new_p, std::map<int, int> &prec_p, int n) {
    
    arma::mat new_data(prec_p.size(),data.n_cols);
    std::map<int, int> old_p;
    int pp = 0;
    for (auto& point : prec_p) {
        new_data.row(pp) = data.row(point.second);
        old_p[pp] = point.second;
        pp++;
    }
    
    arma::mat dist = compute_distances(new_data);
    arma::vec rowSums = arma::zeros(pp);
    
    int idx = new_p.size();
    double max_dist = dist.max();
    for (int i = 0; i < n; ++i) {
        for (int i = 0; i < pp; ++i)
            rowSums(i) = arma::accu(dist.row(i));
        int p = arma::index_min(rowSums);
        dist.row(p).fill(max_dist);
        dist.col(p) = arma::zeros(pp);
        new_p[idx + i] = old_p[p];
        old_p.erase(p);
    }
    
    prec_p = old_p;
    
}

// generate partitions
std::map<int, arma::mat> generate_partitions(arma::mat data, int n_part) {
    
    int n = data.n_rows;
    int d = data.n_cols;
    
    std::map<int, int> first_p;
    for (int i = 0; i < n; ++i)
        first_p[i] = i;
    std::map<int, std::map<int, int>> part_points;
    part_points[0] = first_p;
    
    for (int p = 1; p < n_part; ++p) {
        // create partition from the previous ones
        int n_part = n / (p*(p+1));
        for (int prec_p = 0; prec_p < p; ++prec_p)
            take_n_from_p(data, part_points[p], part_points[prec_p], n_part);
    }

    std::map<int, arma::mat> part_map;
    for (int i=0; i < n_part; ++i) {
        std::map<int, int> all_points = part_points[i];
        arma::mat part(all_points.size(),d);
        int pp = 0;
        for (auto& point : all_points) {
            part.row(pp) = data.row(point.second);
            pp++;
        }
        part_map[i] = part;
    }
    
    return part_map;
}


void save_sol_to_file(arma::mat &X, const char *filename, double ray){
    
    std::ofstream f;
    if (ray == 0)
        f.open(sol_path);
    else {
        std::string file_path = filename;
        auto file = file_path.substr(file_path.find_last_of("/\\") + 1);
        auto name = file.substr(0, file.find_last_of("."));
        int r = ray*100;
        f.open(result_path + name + "_ray_0." + std::to_string(r) + ".txt");
    }
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
    int n_partitions = std::stoi(argv[8]);
    
    arma::mat Ws = read_data(data_path, n, d);
    arma::mat opt_sol = read_sol(opt_path, n, k);
    double opt_mss = compute_mss(Ws, opt_sol);
    arma::mat init_sol;
    if (f > 0) {
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
        save_sol_to_file(init_sol, data_path, 0);
    }
    // arma::mat init_sol = read_sol(sol_path, n, k);
    std::map < int, std::list < std::pair < int, double>>> cls_map;
    double init_mss = compute_clusters(Ws, init_sol, cls_map);
    std::cout << std::endl << "**********************************************************" << std::endl;
    std::cout << "Heuristic MSS " << init_mss << std::endl << std::endl;
    
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
    
    arma::mat sdp_sol;
    double part_mss = 0;
    double sdp_mss = 0;
    UserConstraints constraints;

{
    // solve with different rays
//    double best_ray = -1.0;
//    arma::mat best_sol = init_sol;
//    double best_sol_mss = init_mss;
//    double v_imp;
//    double ray;
//    for (int i=0; i < 5; i++) {
//        ray = 0.85 - i*0.15;
//        log_file << "\nRAY " << ray << ":";
//        std::cout << std::endl << "---------------------------------------------------------------" << std::endl;
//        std::cout << std::endl << "Solving ray " << ray << std::endl;
//        UserConstraints constraints = generate_constraints(cls_map, ray);
//        sdp_mss = sdp_branch_and_bound(k, Ws, constraints, sdp_sol);
//        save_sol_to_file(sdp_sol, data_path, ray);
//        v_imp = (best_sol_mss - sdp_mss) / best_sol_mss;
//        if (v_imp > 0.0000001) {
//            best_ray = ray;
//            best_sol = sdp_sol;
//            best_sol_mss = sdp_mss;
//            std::cout << std::endl << "**********************************************************" << std::endl;
//            std::cout << "Best found!" << std::endl << "Ray " << ray << ". MSS " << best_sol_mss;
//            std::cout << std::endl << "**********************************************************" << std::endl;
//            cls_map = {};
//            compute_clusters(Ws, best_sol, cls_map);
//        }
////        if (v_imp < 0.01) {
////            std::cerr << "Pruning: ray " << ray << ".\n";
////            break;
////        }
//
//    }
    
    
    // solve with n partition
//    std::cout << std::endl << "# Partitions: " << n_partitions << std::endl;
//    std::cout << "---------------------------------------------------------------" << std::endl;
//    for (int i=0; i < n_partitions; ++i) {
//        int part_points = 0;
//        arma::mat part_data(n, d);
//        for (int j = 0; j < k; ++j) {
//            int cls_points = 0;
//            std::list<std::pair<int, double>> points = cls_map[j];
//
//            int n_points = (int) cls_map[j].size()/((double) n_partitions + 1);
//            std::cout << std::endl << "Cluster " << j << " (" <<  cls_map[j].size() <<  ")  - ";
//            std::cout << "Avg points per part: " << n_points << std::endl;
//            auto it = points.begin();
//            if (i == 1)
//                std::advance(it, n_points - 1);
////            else if (i==2)
////                std::advance(it, 3*n_points - 1);
//            std::pair<int, double> p;
//            for (it; it != std::prev(points.end()); ++it) {
//                p = *it;
//                part_data.row(part_points) = Ws.row(p.first);
//                part_points++;
//                cls_points++;
//                if (i == 0 and cls_points >= n_points)
//                    break;
////                else if (i == 1 and cls_points >= n_points)
////                    break;
//            }
//            std::cout << "Added points: " << cls_points << std::endl;
//            log_file << "\nCLUSTER " << j << ": points " << cls_points;
//        }
//        std::cout << std::endl << "Solving" << std::endl;
//        arma::mat part_Ws = part_data.submat(0, 0, part_points - 1, d - 1);
//        sdp_mss = sdp_branch_and_bound(k, part_Ws, constraints, sdp_sol);
//        save_sol_to_file(sdp_sol, data_path, i);
//        std::cout << std::endl << "**********************************************************" << std::endl;
//        std::cout << "Partition " << (i+1) << " Points " << part_points << " MSS " << sdp_mss;
//        std::cout << std::endl << "**********************************************************" << std::endl;
//        part_mss += sdp_mss;
//    }
}
    
    // solve with n uniform partition
    std::cout << std::endl << "# Partitions = " << n_partitions << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    //std::map<int, arma::mat> part_map = generate_partitions(Ws, cls_map, n_partitions);
    std::map<int, arma::mat> part_map = generate_partitions(Ws, n_partitions);
    
    std::cout << std::endl << "Solving" << std::endl;
    for (auto& p : part_map) {
        std::cout << std::endl << "**********************************************************" << std::endl;
        std::cout << "Partition " << (p.first+1) << "\nPoints " << p.second.n_rows;
        std::cout << std::endl << "**********************************************************" << std::endl;
        sdp_mss = sdp_branch_and_bound(k, p.second, constraints, sdp_sol);
        save_sol_to_file(sdp_sol, data_path, p.first);
        part_mss += sdp_mss;
        std::cout << " MSS " << sdp_mss;
    }

    std::cout << std::endl << "**********************************************************" << std::endl;
    std::cout << "Total MSS BOUND " << part_mss << std::endl;
    std::cout << "Heuristic MSS BOUND " << init_mss << std::endl;
    std::cout << "Optimal MSS BOUND " << opt_mss << std::endl;
    std::cout << "GAP Opt " << round((opt_mss - part_mss) / opt_mss * 100) << "%" << std::endl;
    std::cout << "GAP Heur " << round((init_mss - part_mss) / init_mss * 100) << "%" << std::endl;
    std::cout << std::endl << "**********************************************************" << std::endl;

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
