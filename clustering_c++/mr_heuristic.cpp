//
// Created by moden on 08/04/2024.
//
#include "comb_model.h"
#include "kmeans_util.h"
#include "matlab_util.h"
#include "sdp_branch_and_bound.h"

void save_to_file(arma::mat X, std::string name){
    
    std::ofstream f;
    f.open(result_path + "_" + name + ".txt");
    
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

    std::cout << "Added constraints: " << c << std::endl << std::endl;
    
    return constraints;
}


// generate must link constraints on partition sol
int generate_part_constraints(arma::mat sol, int k, UserConstraints &constraints) {
    
    int n = sol.n_rows;
    int d = sol.n_cols - k - 1;
    arma::vec points = sol.col(0);
    arma::mat cls_sol = sol.submat(0, d, n - 1, sol.n_cols - 1);

    int c = 0;
    for (int h = 0; h < k; h++) {
        std::list<int> cls_points = {};
        for (int i = 0; i < n; i++) {
            if (cls_sol(i,h) == 1.0) {
                for (auto& j : cls_points) {
                    std::pair<int,int> ab_pair(points(i),points(j));
                    constraints.ml_pairs.push_back(ab_pair);
                    c++;
                }
                cls_points.push_back(i);
            }
        }
    }
    
    return c;
}


double compute_part_lb(std::map<int, arma::mat> &part_map) {
    
    double lb = 0;
    for (auto& part : part_map) {
        arma::mat dist = compute_distances(part.second);
        lb += accu(dist)/2;
    }
    
    return lb;

}

double compute_comb_bound(arma::mat &data, int p, std::map<int, arma::mat> &sol_map){
    
    int n = data.n_rows;
    int d = data.n_cols;
	double max_dist = 0;
    arma::mat dist = compute_distances(data);

	try {
		GRBEnv *env = new GRBEnv();
		comb_model *model = new comb_gurobi_model(env, n, p);

		model->add_point_constraints();
		model->add_part_constraints();
		//model->add_edge_constraints();
        
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
                    sol_map[h](n_points,0) = i;
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
    
	return max_dist;
}

// generate partitions from clusters
std::map<int, arma::mat> generate_partitions(arma::mat data, int p,
                                             std::map<int, std::list<std::pair<int, double>>> cls_map) {
    
    int n = data.n_rows;
    int d = data.n_cols;
    int k = cls_map.size();

    std::map<int, arma::mat> sol_map;
    for (int h=0; h < p; ++h)
        sol_map[h] = arma::zeros(n, d+1);

    arma::vec n_points = arma::zeros(p);
    for (auto& cls : cls_map) {
        int np = 0;
        int h = 0;
        for (auto point : cls.second) {
//            part = rand() % (n_part);
            sol_map[h](n_points(h),0) = point.first;
            sol_map[h].row(n_points(h)).subvec(1,d) = data.row(point.first);
            n_points(h)++;
            np++;
            if (np > (h + 1)*cls.second.size()/p)
                h++;
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

// compute ub
double compute_ub(arma::mat Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map, int k, int p) {

    std::cout << std::endl << "Generating UB";
    std::cout << std::endl << "*********************************************************************" << std::endl;
    int n_constr = 0;
    UserConstraints part_constraints;
    for (int h = 0; h < p; ++h)
        n_constr += generate_part_constraints(sol_map[h], k, part_constraints);
    std::cout << std::endl << "Added constraints: " << n_constr << std::endl;
    log_file << "Generating UB (added constraints " << n_constr << ")\n";
    double ub_mss = sdp_branch_and_bound(k, Ws, part_constraints, sol);
    std::cout << std::endl << "*********************************************************************" << std::endl;
    std::cout  << std::endl << "UB MSS: " << ub_mss << std::endl;

    return ub_mss;
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
        part.second = std::move(arma::join_horiz(part.second, sol));
    }
    std::cout  << std::endl << std::endl << "LB MSS: " << lb_mss << std::endl;

    return lb_mss;
}

// solve with different rays
double solve_with_ray(arma::mat Ws, arma::mat init_sol, int k, int p, int &num_update, double &lb) {
    
    UserConstraints constraints;
    std::map < int, std::list < std::pair < int, double>>> cls_map;
    double best_mss = compute_clusters(Ws, init_sol, cls_map);
    double sdp_mss;
    double lb_mss;
    double ray;
    int int_ray;
    double best_ray = -1.0;
    arma::mat sdp_sol;
    arma::mat best_sol = init_sol;
    
    std::cout << std::endl << "---------------------------------------------------------------" << std::endl;
    log_file << "--------------------------------------------------------------\n";
    log_file << "Solving with moving ray" << "\n\n";
    for (int i = 0; i < 5; i++) {
        ray = 0.85 - i * 0.15;
        std::cout << std::endl << std::endl;
        std::cout << std::endl << "Solving ray " << ray << std::endl;
        log_file << "Ray " << ray << "\n";
        constraints = generate_constraints(cls_map, ray);
        sdp_mss = sdp_branch_and_bound(k, Ws, constraints, sdp_sol);
        if ((best_mss - sdp_mss) / best_mss > 0.00001) {
            best_ray = ray;
            best_sol = sdp_sol;
            best_mss = sdp_mss;
            num_update++;
            std::cout << std::endl << "**********************************************************" << std::endl;
            std::cout << "Best found!" << std::endl << "Ray " << ray << ". UB MSS " << sdp_mss;
            std::cout << std::endl << "**********************************************************" << std::endl;
            ub_file << "after ray " << std::to_string(ray) << " : " << sdp_mss  << "\n";
            cls_map = {};
            compute_clusters(Ws, best_sol, cls_map);
            std::map<int, arma::mat> sol_map = generate_partitions(Ws, p, cls_map);
            lb_mss = compute_lb(sol_map, k, p);
            if (lb > lb_mss)
                lb = lb_mss;
            lb_file << "after ray " << std::to_string(ray) << " : " << lb_mss << "\n";
            int_ray = ray*100;
            save_to_file(best_sol, "r0" + std::to_string(int_ray));
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
    int ub_update = 0;
    int ray_update = 0;
    double lb_mss;
    double ub_mss;
    double sdp_mss;
    double part_mss;
    arma::mat sdp_sol;
    arma::mat ub_sol;
    arma::mat lb_sol;
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
            // solve with combinatorial bound
            std::cout << "Solving Comb Bound" << std::endl;
            compute_comb_bound(Ws, p, sol_map);
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

        if (part_mss >= lb_mss or it == 0) {
            lb_mss = part_mss;
        }
        lb_sol = sol_map[0];
        for (int h = 1; h < p; ++h)
        	lb_sol = std::move(arma::join_vert(lb_sol, sol_map[h]));
        save_to_file(lb_sol, "LBit" + std::to_string(it));
        
        // create upper bound
        auto start_time_ub = std::chrono::high_resolution_clock::now();
        sdp_mss = compute_ub(Ws, sdp_sol, sol_map, k, p);
        ub_file << "iteration " << std::to_string(it) << " : " << sdp_mss  << "\n";
        auto end_time_ub = std::chrono::high_resolution_clock::now();
        ub_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_ub - start_time_ub).count();
        
        if ((ub_mss - sdp_mss) / ub_mss >= 0.0001 or (it == 0)) {
            ub_mss = sdp_mss;
            best_sol = sdp_sol;
            ub_update++;
        }
        ub_sol = std::move(arma::join_horiz(Ws, sdp_sol));
        ub_sol = std::move(arma::join_horiz(arma::zeros(Ws.n_rows,1), ub_sol));
        save_to_file(ub_sol, "UBit" + std::to_string(it));
        
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

    auto start_time_ray = std::chrono::high_resolution_clock::now();
    sdp_mss = solve_with_ray(Ws, best_sol, k, p, ray_update, lb_mss);
    auto end_time_ray = std::chrono::high_resolution_clock::now();
    if (sdp_mss < ub_mss)
        ub_mss = sdp_mss;


    double ray_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_ray - start_time_ray).count();
    double all_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_ray - start_time_all).count();

    results.ub_mss = ub_mss;
    results.lb_mss = lb_mss;
    results.it = it;
    results.ub_update = ub_update;
    results.ray_update = ray_update;
    results.ub_time = ub_time;
    results.lb_time = lb_time;
    results.ray_time = ray_time;
    results.all_time = all_time;

    return results;
    
}