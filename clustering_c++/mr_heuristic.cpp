//
// Created by moden on 08/04/2024.
//
#include "comb_model.h"
#include "cluster_model.h"
#include "Kmeans.h"
#include "kmeans_util.h"
#include "matlab_util.h"
#include "sdp_branch_and_bound.h"

void save_to_file(arma::mat X, std::string name){
    
    std::ofstream f;
    f.open(result_path + "_" + name + ".txt");
    std::cout << result_path + "_" + name + ".txt" << std::endl << std::endl;

    for (int i = 0; i < X.n_rows; i++){
        double val = X(i,0);
        f << val;
        for (int j = 1; j < X.n_cols; j++){
            val = X(i,j);
            f << " " << val;
        }
        f << "\n";
    }
    f.close();
}

// read lb data
std::map<int, arma::mat> read_part_data(int n, int d, int k, int p) {

    std::ifstream file(sol_path);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    // create sol map
    std::map<int, arma::mat> sol_map;
    arma::vec n_points = arma::zeros(p);
    for (int h=0; h < p; h++)
        sol_map[h] = arma::zeros(n, d+2);

    int part;
    for (int i = 0; i < n; i++) {
        file >> part;
        for (int j = 0; j < d+1; j++)
            file >> sol_map[part-1](n_points(part-1), j);
        n_points(part-1)++;
    }

    for (int h=0; h < p; h++) {
        sol_map[h] = sol_map[h].submat(0, 0, n_points(h) - 1, d);
        if (n_points(h) < k) {
            std::cerr << "read_part_data(): not enough point in partition " << h << " \n";
            exit(EXIT_FAILURE);
        }
    }

    return sol_map;
}

// read lb data
std::map<int, arma::mat> read_part_data2(int n, int d, int k, int p, arma::mat data) {

    std::ifstream file(sol_path);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }

    // create sol map
    std::map<int, arma::mat> sol_map;
    arma::vec n_points = arma::zeros(p);
    for (int h=0; h < p; h++)
        sol_map[h] = arma::zeros(n, d+2);

    int part;
    for (int i = 0; i < n; i++) {
        file >> part;
        sol_map[part](n_points(part), 0) = i+1;
        sol_map[part].row(n_points(part)).subvec(1,d) = data.row(i);
        n_points(part)++;
    }

    for (int h=0; h < p; h++) {
        sol_map[h] = sol_map[h].submat(0, 0, n_points(h) - 1, d);
        if (n_points(h) < k) {
            std::cerr << "read_part_data(): not enough point in partition " << h << " \n";
            exit(EXIT_FAILURE);
        }
    }

    return sol_map;
}

double compute_clusters(arma::mat data, arma::mat sol, std::map<int, std::list<std::pair<int, double>>> &cls_map) {

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
            std::printf("compute_clusters(): cluster %d is empty!\n", j);
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

    std::cout << "Added constraints: " << c << std::endl << std::endl;
    
    return constraints;
}


// generate must link constraints on partition sol
int generate_part_constraints(std::map<int, arma::mat> sol_map, int k, int p, UserConstraints &constraints) {

    int nc = 0;

    for (int h = 0; h < p; h++) {
    	arma::mat sol = sol_map[h];
    	arma::vec point_id = sol.col(0) - 1;
    	arma::vec cls_id = sol.col(sol.n_cols-1);
    	int np = sol.n_rows;

    	for (int c = 0; c < k; c++) {
        	std::list<int> cls_points = {};
        	for (int i = 0; i < np; i++) {
            	if (cls_id(i) == c+1) {
                	for (auto& j : cls_points) {
                    	std::pair<int,int> ab_pair(point_id(i),point_id(j));
                    	constraints.ml_pairs.push_back(ab_pair);
                    	nc++;
                	}
                	cls_points.push_back(i);
            	}
        	}
    	}
    }
    
    return nc;
}


double compute_part_lb(std::map<int, arma::mat> &part_map) {
    
    double lb = 0;
    for (auto& part : part_map) {
        arma::mat dist = compute_distances(part.second);
        lb += accu(dist)/2;
    }
    
    return lb;

}

std::map<int, arma::mat> compute_comb_bound(arma::mat &data, int p){
    
    int n = data.n_rows;
    int d = data.n_cols;
	double max_dist = 0;
    arma::mat dist = compute_distances(data);
    std::map<int, arma::mat> sol_map;

	try {
		GRBEnv *env = new GRBEnv();
		comb_model *model = new comb_gurobi_model(env, n, p);

		model->add_point_constraints();
		model->add_part_constraints();
		model->add_edge_constraints();

        model->set_objective_function(dist);

		model->optimize();
		if(!std::isinf(model->get_value()))
            max_dist = model->get_value();
        
        arma::mat sol = model->get_x_solution();

		// create sol map
        for (int h=0; h < p; ++h) {
        	int n_points = 0;
            sol_map[h] = arma::zeros(n, d+1);
            for (int i=0; i < n; ++i) {
                if (sol(i,h) > 0.9) {
                    sol_map[h](n_points,0) = i+1;
                    sol_map[h].row(n_points).subvec(1,d) = data.row(i);
                    n_points++;
                }
            }
            sol_map[h] = sol_map[h].submat(0, 0, n_points - 1, d);
        }
        
		delete model;
		delete env;
    
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    
    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "Combinatorial BOUND OBJ " << max_dist << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;
    
	return sol_map;
}

std::map<int, arma::mat> compute_cluster_bound(arma::mat &data, int p){

    int n = data.n_rows;
    int d = data.n_cols;
    int k = init_sol.n_cols;
	double max_dist = 0;
    arma::mat dist = compute_distances(data);
    std::map<int, arma::mat> sol_map;

	try {
		GRBEnv *env = new GRBEnv();
		cluster_model *model = new cluster_gurobi_model(env, n, p, k, dist);

		model->add_point_constraints();
		model->add_part_constraints();
		model->add_edge_constraints();
		model->add_min_constraints(dist);

		model->optimize();
		if(!std::isinf(model->get_value()))
            max_dist = model->get_value();

        arma::mat sol = model->get_x_solution();

		// create sol map
        for (int h=0; h < p; ++h) {
        	int n_points = 0;
            sol_map[h] = arma::zeros(n, d+1);
            for (int i=0; i < n; ++i) {
                if (sol(i,h) > 0.9) {
                    sol_map[h](n_points,0) = i+1;
                    sol_map[h].row(n_points).subvec(1,d) = data.row(i);
                    n_points++;
                }
            }
            sol_map[h] = sol_map[h].submat(0, 0, n_points - 1, d);
        }

		delete model;
		delete env;

    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }

    std::cout << "---------------------------------------------------------------" << std::endl;
    std::cout << "Cluster BOUND OBJ " << max_dist << std::endl;
    std::cout << "---------------------------------------------------------------" << std::endl;

	return sol_map;
}

// generate partitions from clusters
std::map<int, arma::mat> generate_partitions(arma::mat data, int p,
                                             std::map<int, std::list<std::pair<int, double>>> cls_map) {
    
    int n = data.n_rows;
    int d = data.n_cols;
    int k = cls_map.size();
    arma::vec n_points = arma::zeros(p);

    int part;
    std::map<int, arma::mat> sol_map;
    for (int h=0; h < p; ++h)
        sol_map[h] = arma::zeros(n, d+1);
    
    // partition points by clusters
    if (part_m == 'c') {
    	for (auto& cls : cls_map) {
        	int np = 0;
        	part = 0;
        	for (auto point : cls.second) {
            	sol_map[part](n_points(part),0) = point.first+1;
            	sol_map[part].row(n_points(part)).subvec(1,d) = data.row(point.first);
            	n_points(part)++;
            	np++;
            	if (np > (part + 1)*cls.second.size()/p)
                	part++;
        	}
        }
    }

    // random point to partitions
    else if (part_m == 'r' or part_m == 'f') {
    	for (int i=0; i < n; ++i) {
        	do {
            	part = rand() % (p); // Generate a new random part
        	} while (n_points(part) >= std::floor(n/p) + 1);
        	sol_map[part](n_points(part),0) = i+1;
        	sol_map[part].row(n_points(part)).subvec(1,d) = data.row(i);
        	n_points(part)++;
    	}
    }

    for (int h=0; h < p; ++h)
        sol_map[h] = sol_map[h].submat(0, 0, n_points(h) - 1, d);
    
    return sol_map;
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
        int p = arma::index_max(rowSums);
        dist.row(p).fill(0);
//        dist.row(p).fill(max_dist);
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
        int n_point = n / n_part;
//        int n_part = n / (p*(p+1));
//        for (int prec_p = 0; prec_p < p; ++prec_p)
//            take_n_from_p(data, part_points[p], part_points[prec_p], n_part);)
        take_n_from_p(data, part_points[p], part_points[0], n_point);

//        for (int prec_p = 0; prec_p < p; ++prec_p) {
//            if (prec_p==0)
//                take_n_from_p(data, part_points[p], part_points[prec_p], 30);
//            else
//                take_n_from_p(data, part_points[p], part_points[prec_p], 10);
//        }
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
        std::cout << "Partition: " << i << " points = " << part.n_rows << std::endl;
    }
    
    return part_map;
}

// compute lb
double compute_lb(std::map<int, arma::mat> &sol_map, int k, int p) {

    std::cout << std::endl << "Generating LB";
    double lb_mss = 0;
    UserConstraints constraints;
    for (auto &part: sol_map) {
        int np = part.second.n_rows;
        int d = part.second.n_cols - 1;
        std::cout << std::endl << "*********************************************************************" << std::endl;
        std::cout << "Partition " << (part.first + 1) << "\nPoints " << np;
        std::cout << std::endl << "*********************************************************************" << std::endl;
        log_file << "Partition " << (part.first + 1) << "\n";
        arma::mat data = part.second.submat(0, 1, np-1, d);
        arma::mat sol(np,k);
        lb_mss += sdp_branch_and_bound(k, data, constraints, sol);
        arma::mat cls(np,1);
        for (int i = 0; i < np; i++)
            for (int c = 0; c < k; c++)
                if (sol(i,c)==1)
                    cls(i)= c+1;
        part.second = std::move(arma::join_horiz(part.second, cls));
    }
    std::cout  << std::endl << std::endl << "LB MSS: " << lb_mss << std::endl;
    log_file << "Merge LB MSS: " << lb_mss << "\n\n\n";

    return lb_mss;
}

// compute ub
double compute_ub(arma::mat Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map, int k, int p) {

    std::cout << std::endl << "Generating UB";
    std::cout << std::endl << "*********************************************************************" << std::endl;
    UserConstraints part_constraints;
    int n_constr = generate_part_constraints(sol_map, k, p, part_constraints);
    std::cout << std::endl << "Added constraints: " << n_constr << std::endl;
    log_file << "Generating UB (added constraints " << n_constr << ")\n";
    double ub_mss = sdp_branch_and_bound(k, Ws, part_constraints, sol);
    std::cout << std::endl << "*********************************************************************" << std::endl;
    std::cout  << std::endl << "UB MSS: " << ub_mss << std::endl;

    return ub_mss;
}

// compute lb sol by merging partitions sol
arma::mat save_lb(std::map<int, arma::mat> &sol_map, int p){

	arma::mat part = arma::ones(sol_map[0].n_rows,1);
    arma::mat sol = std::move(arma::join_horiz(part, sol_map[0]));
    for (int h = 1; h < p; ++h) {
        part = arma::vec(sol_map[h].n_rows,1).fill(h+1);
        arma::mat solp = std::move(arma::join_horiz(part, sol_map[h]));
        sol = std::move(arma::join_vert(sol, solp));
    }

    return sol;
}

arma::mat save_ub(arma::mat data, arma::mat sol) {

	int n = data.n_rows;
	int k = sol.n_cols;
    arma::mat cls(sol.n_rows,1);
    arma::mat id(sol.n_rows,2);
    for (int i = 0; i < data.n_rows; i++) {
        for (int c = 0; c < k; c++) {
        	if (sol(i,c)==1) {
        		cls(i) = c+1;
        		id(i,0) = 0;
        		id(i,1) = i+1;
        	}
        }
    }
    arma::mat ub_sol = std::move(arma::join_horiz(id, data));
    ub_sol = std::move(arma::join_horiz(ub_sol, cls));

    return ub_sol;

}

// solve with different rays
double solve_with_ray(arma::mat Ws, arma::mat init_sol, int k, int p, int &ub_update, int &lb_update, double &lb) {
    
    UserConstraints constraints;
    std::map < int, std::list < std::pair < int, double>>> cls_map;
    double best_mss = compute_clusters(Ws, init_sol, cls_map);
    double lb_mss;
    double ub_mss;
    double ray;
    int int_ray;
    double best_ray = -1.0;
    arma::mat sdp_sol;
    arma::mat best_sol = init_sol;

    std::cout << std::endl << "---------------------------------------------------------------" << std::endl;
    log_file << "--------------------------------------------------------------\n";
    log_file << "Solving with moving ray" << "\n\n";
    for (int i = 0; i < 0; i++) {
        ray = 0.85 - i * 0.15;
        int_ray = ray*100;
        std::cout << std::endl << std::endl;
        std::cout << std::endl << "Solving ray " << ray << std::endl;
        log_file << "Ray " << ray << "\n";
        constraints = generate_constraints(cls_map, ray);
        ub_mss = sdp_branch_and_bound(k, Ws, constraints, sdp_sol);
        save_to_file(save_ub(Ws, sdp_sol), "UBr0" + std::to_string(int_ray));
        if ((best_mss - ub_mss) / best_mss > 0.00001) {
            best_ray = ray;
            best_sol = sdp_sol;
            best_mss = ub_mss;
            ub_update++;
            std::cout << std::endl << "**********************************************************" << std::endl;
            std::cout << "Best UB found!" << std::endl << "Ray " << ray << ". UB MSS " << best_mss;
            std::cout << std::endl << "**********************************************************" << std::endl;
            ub_file << "after ray " << std::to_string(ray) << " : " << best_mss  << "\n";
            cls_map = {};
            compute_clusters(Ws, best_sol, cls_map);
            std::map<int, arma::mat> sol_map = generate_partitions(Ws, p, cls_map);
            lb_mss = compute_lb(sol_map, k, p);
            if (lb_mss > lb) {
                lb = lb_mss;
                lb_update++;
            	std::cout << std::endl << "**********************************************************" << std::endl;
            	std::cout << "Best LB found!" << std::endl << "LB MSS " << lb_mss;
            	std::cout << std::endl << "**********************************************************" << std::endl;
            }
            lb_file << "after ray " << std::to_string(ray) << " : " << lb_mss << "\n";
        	save_to_file(save_lb(sol_map, p), "LBr0" + std::to_string(int_ray));
        }
//        if (v_imp < 0.01) {
//            std::cerr << "Pruning: ray " << ray << ".\n";
//            break;
//        }
    }
    
    return best_mss;
    
}


ResultData mr_heuristic(int k, int p, arma::mat Ws) {

    ResultData results;

    int it = 0;
    int lb_update = -1;
    int ub_update = -1;
    int ray_lb_update = 0;
    int ray_ub_update = 0;
    double lb_mss;
    double ub_mss;
    double sdp_mss;
    double part_mss;
    arma::mat sdp_sol;
    arma::mat best_sol;
    UserConstraints constraints;
    std::map<int, arma::mat> sol_map;
    
    bool improvement = true;
    auto start_time_all = std::chrono::high_resolution_clock::now();
    double ub_time = 0;
    double lb_time = 0;

    while (improvement) {
    
        std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;
        std::cout << "It " << it << std::endl;
        log_file << "---------------------------------------------------------------\n";
        log_file << "It " << it << "\n";
        if (it == 0) {
            if (part_m == 'c') {
                std::cout << "Solving Comb Bound" << std::endl;
                sol_map = compute_comb_bound(Ws, p);
            }
            else if (part_m == 'f') {
                std::cout << "Loading part from file" << std::endl;
                sol_map = read_part_data(Ws.n_rows, Ws.n_cols, k, p);
            }
            else {
                std::cout << "Generating randomly" << std::endl;
                std::map < int, std::list < std::pair < int, double>>> cls_map;
                sol_map = generate_partitions(Ws, p, cls_map);
            }
        }
        else{
            // solve with n partition from cluster
            std::map < int, std::list < std::pair < int, double>>> cls_map;
            compute_clusters(Ws, sdp_sol, cls_map);
            sol_map = generate_partitions(Ws, p, cls_map);
        }


        // generating lb
        auto start_time_lb = std::chrono::high_resolution_clock::now();
        part_mss = compute_lb(sol_map, k, p);
        lb_file << "iteration " << std::to_string(it) << " : " << part_mss  << "\n";
        auto end_time_lb = std::chrono::high_resolution_clock::now();
        lb_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

        if (part_mss > lb_mss or it == 0) {
            lb_mss = part_mss;
            lb_update++;
        }
        save_to_file(save_lb(sol_map, p), "LBit" + std::to_string(it));
        
        // create upper bound
        auto start_time_ub = std::chrono::high_resolution_clock::now();
        sdp_mss = compute_ub(Ws, sdp_sol, sol_map, k, p);
        ub_file << "iteration " << std::to_string(it) << " : " << sdp_mss  << "\n";
        auto end_time_ub = std::chrono::high_resolution_clock::now();
        ub_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_ub - start_time_ub).count();
        save_to_file(save_ub(Ws, sdp_sol), "UBit" + std::to_string(it));
        
        if ((ub_mss - sdp_mss) / ub_mss >= 0.0001 or (it == 0)) {
            ub_mss = sdp_mss;
            best_sol = sdp_sol;
            ub_update++;
        }
        
        std::cout << std::endl << std::endl << "--------------------------------------------------------------------";
        std::cout << std::endl << "It " << it << " GAP UB-LB " << round((ub_mss - lb_mss) / ub_mss * 100) << "%" << std::endl;
        std::cout << "--------------------------------------------------------------------" << std::endl;

        if (part_mss < lb_mss)
            improvement = false;

        it++;

    }
    
    std::cout << std::endl << "*********************************************************************" << std::endl;
    std::cout  << std::endl << "Best UB MSS: " << ub_mss << std::endl;
    std::cout << std::endl << "*********************************************************************" << std::endl;

    log_file << "\n\n--------------------------------------------------------------------\n";
    log_file << "Final GAP UB-LB " << round((ub_mss - lb_mss) / ub_mss * 100) << "%";
    log_file << "\n--------------------------------------------------------------------\n";

    auto start_time_ray = std::chrono::high_resolution_clock::now();
    sdp_mss = solve_with_ray(Ws, best_sol, k, p, ray_ub_update, ray_lb_update, lb_mss);
    auto end_time_ray = std::chrono::high_resolution_clock::now();
    if (sdp_mss < ub_mss)
        ub_mss = sdp_mss;


    double ray_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_ray - start_time_ray).count();
    double all_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_ray - start_time_all).count();

    results.it = it;
    results.lb_mss = lb_mss;
    results.ub_mss = ub_mss;
    results.lb_update = lb_update;
    results.ub_update = ub_update;
    results.ray_lb_update = ray_lb_update;
    results.ray_ub_update = ray_ub_update;
    results.lb_time = lb_time;
    results.ub_time = ub_time;
    results.ray_time = ray_time;
    results.all_time = all_time;

    return results;
    
}


ResultData mr_heuristic_only_ray(int k, int p, arma::mat Ws) {

    ResultData results;

    int it = 0;
    int lb_update = -1;
    int ub_update = -1;
    int ray_lb_update = 0;
    int ray_ub_update = 0;
    double lb_mss;
    double ub_mss;
    arma::mat sdp_sol;
    UserConstraints constraints;
    std::map<int, arma::mat> sol_map;

    bool improvement = true;
    auto start_time_all = std::chrono::high_resolution_clock::now();
    double ub_time = 0;
    double lb_time = 0;

    std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;
    std::cout << "Comb bound" << std::endl;

    std::cout << "Solving Comb Bound" << std::endl;
    sol_map = compute_comb_bound(Ws, p);

    // generating lb
    auto start_time_lb = std::chrono::high_resolution_clock::now();
    lb_mss = compute_lb(sol_map, k, p);
    lb_file << "iteration " << std::to_string(it) << " : " << lb_mss  << "\n";
    auto end_time_lb = std::chrono::high_resolution_clock::now();
    lb_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();
    sdp_sol = save_lb(sol_map, p);
    save_to_file(sdp_sol, "LBit" + std::to_string(it));

    // create upper bound
    auto start_time_ub = std::chrono::high_resolution_clock::now();
    ub_mss = compute_ub(Ws, sdp_sol, sol_map, k, p);
    ub_file << "iteration " << std::to_string(it) << " : " << ub_mss  << "\n";
    auto end_time_ub = std::chrono::high_resolution_clock::now();
    ub_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_ub - start_time_ub).count();
    save_to_file(save_ub(Ws, sdp_sol), "UBit" + std::to_string(it));

    std::cout << std::endl << std::endl << "--------------------------------------------------------------------";
    std::cout << std::endl << "It " << it << " GAP UB-LB " << round((ub_mss - lb_mss) / ub_mss * 100) << "%" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    log_file << "\n\n--------------------------------------------------------------------\n";
    log_file << "Final GAP UB-LB " << round((ub_mss - lb_mss) / ub_mss * 100) << "%";
    log_file << "\n--------------------------------------------------------------------\n";

    auto start_time_ray = std::chrono::high_resolution_clock::now();
    ub_mss = solve_with_ray(Ws, sdp_sol, k, p, ray_ub_update, ray_lb_update, lb_mss);
    auto end_time_ray = std::chrono::high_resolution_clock::now();

    double ray_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_ray - start_time_ray).count();
    double all_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_ray - start_time_all).count();

    results.lb_mss = lb_mss;
    results.ub_mss = ub_mss;
    results.it = it;
    results.lb_update = lb_update;
    results.ub_update = ub_update;
    results.ray_lb_update = ray_lb_update;
    results.ray_ub_update = ray_ub_update;
    results.lb_time = lb_time;
    results.ub_time = ub_time;
    results.ray_time = ray_time;
    results.all_time = all_time;

    return results;

}

std::pair<double, double> test_lb(arma::mat Ws, int p, int k) {

    double lb_mss;
    double ub_mss;
    arma::mat sdp_sol;
    UserConstraints constraints;
    std::map<int, arma::mat> sol_map;

    std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;

    // generating lb
    int n = Ws.n_rows;
    int d = Ws.n_cols;
    if (part_m == 'p')
        sol_map = compute_cluster_bound(Ws, p);
    if (part_m == 'c')
        sol_map = compute_comb_bound(Ws, p);
    if (part_m == 'r') {
        std::map < int, std::list < std::pair < int, double>>> cls_map;
        sol_map = generate_partitions(Ws, p, cls_map);
    }
    if (part_m == 'f') {
        sol_map = read_part_data2(n, d, k, p, Ws);
    }
    if (part_m == 'k') {
        arma::mat cent = arma::repmat(arma::mean(Ws, 0), n, 1);
        arma::mat dist = arma::sqrt(arma::sum(arma::square(Ws - cent), 1));
        arma::mat new_dist = dist.max() - dist;
        arma::mat Ws2 = Ws.each_col() + new_dist*0.5;

        arma::mat sol;
        std::map<int, std::set<int>> ml_map = {};
        std::vector <std::pair<int, int>> local_cl = {};
        std::vector <std::pair<int, int>> global_ml = {};
        std::vector <std::pair<int, int>> global_cl = {};
        Kmeans kmeans(Ws2, p, ml_map, local_cl, global_ml, global_cl, 0);
        kmeans.start(100, 100, 5);
        sol = kmeans.getAssignments();
        for (int h=0; h < p; ++h) {
            int n_points = 0;
            sol_map[h] = arma::zeros(n, d+1);
            for (int i=0; i < n; ++i) {
                if (sol(i,h) == 1) {
                    sol_map[h](n_points,0) = i+1;
                    sol_map[h].row(n_points).subvec(1,d) = Ws.row(i);
                    n_points++;
                }
            }
            sol_map[h] = sol_map[h].submat(0, 0, n_points - 1, d);
        }
    }
    if (part_m == 'a') {

        arma::vec ncls(k);
        arma::vec cls(n);
		arma::mat distances = arma::zeros(n, n);
		for (int i = 0; i < n; i++) {
        	for (int c=0; c < k; ++c) {
				if (init_sol(i,c) == 1) {
        			ncls(c)++;
        			cls(i) = c;
        		}
        	}
			for (int j = i+1; j < n; j++) {
        		for (int c=0; c < k; ++c) {
        			 if (init_sol(i,c) == 1 and init_sol(j,c) == 1) {
						distances(i, j) = squared_distance(Ws.row(i).t(), Ws.row(j).t());
						distances(j, i) = squared_distance(Ws.row(i).t(), Ws.row(j).t());
					}
				}
			}
		}

        /* Start main iteration loop for exchange procedure */
        std::map<int, std::list<int>> best_part_map;
        double best_dist = 0;

        for (int l = 0; l < 1000; l++) {

        	std::map<int, std::list<int>> part_map;
        	arma::vec part = arma::zeros(n) - 1;
            for (int h = 0; h < p; h++)
            	part_map[h] = {};

        	// random point per cluster to partitions
        	int h;
        	arma::vec part_dist(p);
        	arma::mat count(k, p);
        	double dist = 0;
        	for (int i=0; i < n; ++i) {
        		int c = cls(i);
        		int nc = ncls(c);
            	do {
                	h = rand() % (p);
            	} while (count(c, h) >= std::floor(nc/p) + 1);
            	part(i) = h;
            	for (int j : part_map[h]) {
                	part_dist(h) += distances(i,j);
            		dist += distances(i,j);
            	}
            	part_map[h].push_back(i);
            	count(c, h)++;
        	}

        	if (l == 0) {
				best_part_map = part_map;
				best_dist = dist;
        	}

        /* 1. Level: Iterate through `n` data points */
        for (int i = 0; i < n-1; i++) {
            double best_obj = dist;
            int h1 = part(i);

            // Initialize `best` variable for the i'th item
            double dist_h1;
            double dist_h2;
            double part_dist_h1 = 0;
            double part_dist_h2 = 0;
            std::pair<int, int> best_swap(NULL,NULL);

            /* 2. Level: Iterate through the exchange partners */
            for (int j = i+1; j < n; j++) {

            /* only same cluster */
                if (part(j) != h1 and cls(i) == cls(j)) {

            		double swap_obj = 0;

					int h2 = part(j);
					dist_h1 = part_dist(h1);
					dist_h2 = part_dist(h2);

                    // Update objective
                    for (int h3 = 0; h3 < p; h3++)
                        if (h3 != h1 and h3 != h2)
                            swap_obj += part_dist(h3);

                    // Partition h1: Loses distances to element i and gains j
            		for (int s : part_map[h1]) {
                		dist_h1 -= distances(i,s);
                		dist_h1 += distances(s,j);
                	}
                    swap_obj += dist_h1;

                    // Partition h2: Loses distances to element j and gains i
            		for (int s : part_map[h2]) {
                		dist_h2 -= distances(j,s);
                		dist_h2 += distances(s,i);
                	}
                    swap_obj += dist_h2;

                    // Update `best` if objective was improved
                    if (swap_obj > best_obj) {
                        best_obj = swap_obj;
                        part_dist_h1 = dist_h1;
                        part_dist_h2 = dist_h2;
                        best_swap = std::pair<int, int>(i, j);
                    }
                }
            }

            // Only if objective is improved: Do the swap
            if (best_obj > dist) {
                dist = best_obj;
                int a = best_swap.first;
                int b = best_swap.second;
                int h2 = part(b);
                part_map[h1].remove(a);
                part_map[h1].push_back(b);
                part_map[h2].remove(b);
                part_map[h2].push_back(a);
                part(a) = h2;
                part(b) = h1;
                part_dist(h1) = part_dist_h1;
                part_dist(h2) = part_dist_h2;
            }

        	if (dist > best_dist) {
        		best_dist = dist;
        		best_part_map = part_map;
        	}
        }
    }

        // create sol map
        for (int h=0; h < p; ++h) {
            sol_map[h] = arma::zeros(best_part_map[h].size(), d+1);
            int np = 0;
            for (int i : best_part_map[h]) {
                sol_map[h](np,0) = i+1;
                sol_map[h].row(np).subvec(1,d) = Ws.row(i);
                np++;
            }
        }

        for (int h = 0; h < p; h++) {
            std::cout << "part " << h << ": ";
            arma::vec countc(k);
            for (auto& i : best_part_map[h]) {
                std::cout << i;
                countc(cls(i))++;
            }
            std::cout << std::endl;
            for (int c = 0;  c<k ; c++)
                std::cout << "num cluster" << countc(c) << std::endl;
        }

        std::cout << "Best " << best_dist << std::endl;

    }

    if (part_m == 'v') {

        arma::vec ncls(k);
        arma::vec cls(n);
		arma::mat distances = arma::zeros(n, n);
		for (int i = 0; i < n; i++) {
        	for (int c=0; c < k; ++c) {
				if (init_sol(i,c) == 1) {
        			ncls(c)++;
        			cls(i) = c;
        		}
        	}
			for (int j = i+1; j < n; j++) {
        		for (int c=0; c < k; ++c) {
        			 if (init_sol(i,c) == 1 and init_sol(j,c) == 1) {
						distances(i, j) = squared_distance(Ws.row(i).t(), Ws.row(j).t());
						distances(j, i) = squared_distance(Ws.row(i).t(), Ws.row(j).t());
					}
				}
			}
		}

        /* Start main iteration loop for exchange procedure */
        std::map<int, std::list<int>> best_part_map;
        double best_dist = 0;

        std::map<int, std::list<int>> part_map;
        arma::vec part = arma::zeros(n) - 1;
        for (int h = 0; h < p; h++)
            part_map[h] = {};

        arma::vec part_dist(p);
        arma::mat count(k, p);

        double dist = 0;
        std::ifstream file(sol_path);
        if (!file) {
            std::cerr << strerror(errno) << "\n";
            exit(EXIT_FAILURE);
        }
        int i;
        int h;
        for (int r=0; r < n; ++r) {
            file >> h;
            h--;
            file >> i;
            i--;
            part(i) = h;
            for (int j : part_map[h]) {
                part_dist(h) += distances(i,j);
            	dist += distances(i,j);
            }
            part_map[h].push_back(i);
            count(cls(i), part(i))++;
            for (int j = 0; j < d + 1; j++)
            	file >> i;
        }

        best_part_map = part_map;
		best_dist = dist;

        /* 1. Level: Iterate through `n` data points */
        for (int i = 0; i < n-1; i++) {
            double best_obj = dist;
            int h1 = part(i);

            // Initialize `best` variable for the i'th item
            double dist_h1;
            double dist_h2;
            double part_dist_h1 = 0;
            double part_dist_h2 = 0;
            std::pair<int, int> best_swap(NULL,NULL);

            /* 2. Level: Iterate through the exchange partners */
            for (int j = i+1; j < n; j++) {

            /* only same cluster */
                if (part(j) != h1 and cls(i) == cls(j)) {

            		double swap_obj = 0;

					int h2 = part(j);
					dist_h1 = part_dist(h1);
					dist_h2 = part_dist(h2);

                    // Update objective
                    for (int h3 = 0; h3 < p; h3++)
                        if (h3 != h1 and h3 != h2)
                            swap_obj += part_dist(h3);

                    // Partition h1: Loses distances to element i and gains j
            		for (int s : part_map[h1]) {
                		dist_h1 -= distances(i,s);
                		dist_h1 += distances(s,j);
                	}
                    swap_obj += dist_h1;

                    // Partition h2: Loses distances to element j and gains i
            		for (int s : part_map[h2]) {
                		dist_h2 -= distances(j,s);
                		dist_h2 += distances(s,i);
                	}
                    swap_obj += dist_h2;

                    // Update `best` if objective was improved
                    if (swap_obj > best_obj) {
                        best_obj = swap_obj;
                        part_dist_h1 = dist_h1;
                        part_dist_h2 = dist_h2;
                        best_swap = std::pair<int, int>(i, j);
                    }
                }
            }

            // Only if objective is improved: Do the swap
            if (best_obj > dist) {
                dist = best_obj;
                int a = best_swap.first;
                int b = best_swap.second;
                int h2 = part(b);
                part_map[h1].remove(a);
                part_map[h1].push_back(b);
                part_map[h2].remove(b);
                part_map[h2].push_back(a);
                part(a) = h2;
                part(b) = h1;
                part_dist(h1) = part_dist_h1;
                part_dist(h2) = part_dist_h2;
                std::cout << "BEST FOUND" << std::endl;
                std::cout << cls(a) << " and " << cls(b) << std::endl;
            }

        	if (dist > best_dist) {
        		best_dist = dist;
        		best_part_map = part_map;
        	}

        }

        for (int h = 0; h < p; h++) {
            std::cout << "part " << h << ": ";
            arma::vec countc(k);
            for (auto& i : best_part_map[h]) {
                std::cout << i;
                countc(cls(i))++;
            }
            std::cout << std::endl;
            for (int c = 0;  c<k ; c++)
                std::cout << "num cluster" << countc(c) << std::endl;
        }

        std::cout << "Best " << best_dist << std::endl;
        // create sol map
        for (int h=0; h < p; ++h) {
            sol_map[h] = arma::zeros(best_part_map[h].size(), d+1);
            int np = 0;
            for (int i : best_part_map[h]) {
                sol_map[h](np,0) = i+1;
                sol_map[h].row(np).subvec(1,d) = Ws.row(i);
                np++;
            }
        }

    }

    // create lower bound
    lb_mss = compute_lb(sol_map, k, p);
    sdp_sol = save_lb(sol_map, p);
    save_to_file(sdp_sol, "LB_method_" + std::string(1,part_m));

    // create upper bound
    ub_mss = compute_ub(Ws, sdp_sol, sol_map, k, p);
    sdp_sol = save_ub(Ws, sdp_sol);
    save_to_file(sdp_sol, "UB_method_" + std::string(1,part_m));
    std::cout << ub_mss  << "\n";

    double gap = round((ub_mss - lb_mss) / ub_mss * 100);
    std::cout << std::endl << std::endl << "--------------------------------------------------------------------";
    std::cout << std::endl << "Method " << part_m << " GAP UB-LB " << gap << "%" << std::endl;
    std::cout << "--------------------------------------------------------------------" << std::endl;

    return std::pair<double, double>(lb_mss, gap);

}