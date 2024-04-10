//
// Created by moden on 08/04/2024.
//
#include "comb_model.h"
#include "kmeans_util.h"
#include "matlab_util.h"
#include "sdp_branch_and_bound.h"

void save_to_file(arma::mat &X, std::string file, std::string name){
    
    std::ofstream f;
    f.open(file + "_" + name + ".txt");
    
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

    std::cout << "Added constraints: " << c << std::endl << std::endl;
    
    return constraints;
}


// generate must link constraints on partition sol
int generate_part_constraints(arma::mat sol, UserConstraints &constraints, arma::vec points) {
    
    int n = sol.n_rows;
    int k = sol.n_cols;
    
    int c = 0;
    for (int h = 0; h < k; h++) {
        std::list<int> cls_points = {};
        for (int i = 0; i < n; i++) {
            if (sol(i,h) == 1.0) {
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

double compute_comb_bound(arma::mat &data, int p, std::map<int, arma::mat> &part_map, std::map<int, arma::vec> &point_map){
    
    int n = data.n_rows;
    int d = data.n_cols;
    arma::mat dist = compute_distances(data);
	double max_dist = 0;

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
        
        arma::vec n_points = arma::zeros(p);
        for (int i=0; i < p; ++i) {
            part_map[i] = arma::zeros(n, d);
            point_map[i] = arma::zeros(n);
        }
        
        for (int h = 0; h < p; ++h) {
            for (int i=0; i < n; ++i) {
                if (sol(i,h) > 0.9) {
                    part_map[h].row(n_points(h)) = data.row(i);
                    point_map[h](n_points(h)) = i;
                    n_points(h)++;
                }
            }
        }
        
        for (int i=0; i < p; ++i) {
            part_map[i] = part_map[i].submat(0, 0, n_points(i) - 1, d - 1);
            point_map[i] = point_map[i].subvec(0, n_points(i) - 1);
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
std::map<int, arma::mat> generate_partitions(arma::mat data,
                                             std::map<int, std::list<std::pair<int, double>>> cls_map, int n_part,
                                             std::map<int, arma::vec> &point_map) {
    
    int n = data.n_rows;
    int d = data.n_cols;
    int k = cls_map.size();
    
    std::map<int, arma::mat> part_map;
    arma::vec n_points = arma::zeros(n_part);
    for (int i=0; i < n_part; ++i) {
        part_map[i] = arma::zeros(n, d);
        point_map[i] = arma::zeros(n);
    }
    
    std::list<std::pair<int, double>> points;
    for (auto& cls : cls_map) {
        int np = 0;
        int part = 0;
        int part_points = n_points(part);
        for (auto p : cls.second) {
//            part = rand() % (n_part);
            part_map[part].row(n_points(part)) = data.row(p.first);
            point_map[part](n_points(part)) = p.first;
            n_points(part)++;
            np++;
            if (np > (part + 1)*cls.second.size()/n_part)
                part++;
        }
    }

    for (int i=0; i < n_part; ++i) {
        part_map[i] = part_map[i].submat(0, 0, n_points(i) - 1, d - 1);
        point_map[i] = point_map[i].subvec(0, n_points(i) - 1);
    }
    
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

// solve with different rays
double solve_with_ray(arma::mat Ws, arma::mat init_sol, int k, std::string result_path) {
    
    UserConstraints constraints;
    std::map < int, std::list < std::pair < int, double>>> cls_map;
    double best_mss = compute_clusters(Ws, init_sol, cls_map);
    double sdp_mss;
    double ray;
    double best_ray = -1.0;
    arma::mat sdp_sol;
    arma::mat best_sol = init_sol;
    
    std::cout << std::endl << "---------------------------------------------------------------" << std::endl;
    for (int i = 0; i < 4; i++) {
        ray = 0.85 - i * 0.15;
        std::cout << std::endl << std::endl;
        std::cout << std::endl << "Solving ray " << ray << std::endl;
        constraints = generate_constraints(cls_map, ray);
        sdp_mss = sdp_branch_and_bound(k, Ws, constraints, sdp_sol);
        if ((best_mss - sdp_mss) / best_mss > 0.00001) {
            best_ray = ray;
            best_sol = sdp_sol;
            best_mss = sdp_mss;
            std::cout << std::endl << "**********************************************************" << std::endl;
            std::cout << "Best found!" << std::endl << "Ray " << ray << ". UB MSS " << sdp_mss;
            std::cout << std::endl << "**********************************************************" << std::endl;
            cls_map = {};
            compute_clusters(Ws, best_sol, cls_map);
            save_to_file(best_sol, result_path, "ray0" + std::to_string((int) ray*100));
        }
//        if (v_imp < 0.01) {
//            std::cerr << "Pruning: ray " << ray << ".\n";
//            break;
//        }
    }
    
    return best_mss;
    
}


std::pair<double,double> mr_heuristic(int k, int p, arma::mat Ws, std::string result_path, int it) {
    
    double lb_mss;
    double ub_mss;
    double sdp_mss;
    double part_mss;
    arma::mat sdp_sol;
    arma::mat ub_sol;
    arma::mat best_sol;
    UserConstraints constraints;
    std::map<int, arma::mat> part_map;
    std::map<int, arma::vec> point_map;
    std::map<int, arma::mat> sol_map;
    
    bool improvement = true;
    
    while (improvement) {
    
        std::cout << std::endl << "--------------------------------------------------------------------" << std::endl;
        std::cout << "It " << it << std::endl;
        if (it == 0) {
            // solve with combinatorial bound
            std::cout << "Solving Comb bound:" << std::endl;
            compute_comb_bound(Ws, p, part_map, point_map);
        }
        else{
            // solve with n partition from cluster
            std::map < int, std::list < std::pair < int, double>>> cls_map;
            compute_clusters(Ws, ub_sol, cls_map);
            part_map = generate_partitions(Ws, cls_map, p, point_map);
        }
        
        std::cout << std::endl << "Generating LB";
        part_mss = 0;
        for (auto &p: part_map) {
            std::cout << std::endl << "*********************************************************************" << std::endl;
            std::cout << "Partition " << (p.first + 1) << "\nPoints " << p.second.n_rows;
            std::cout << std::endl << "*********************************************************************" << std::endl;
            part_mss += sdp_branch_and_bound(k, p.second, constraints, sdp_sol);
            sol_map[p.first] = sdp_sol;
        }
        std::cout  << std::endl << std::endl << "LB MSS: " << part_mss << std::endl;
        
        if (it == 0)
            lb_mss = part_mss;
        if (part_mss >= lb_mss) {
            lb_mss = part_mss;
            for (int i = 0; i < p; ++i)
                save_to_file(sol_map[i], result_path, "part" + std::to_string(i));
        }
        
        // create upper bound
        std::cout << std::endl << "Generating UB";
        std::cout << std::endl << "*********************************************************************" << std::endl;
        int n_constr = 0;
        UserConstraints part_constraints;
        for (int i = 0; i < p; ++i)
            n_constr += generate_part_constraints(sol_map[i], part_constraints, point_map[i]);
        std::cout << std::endl << "Added constraints: " << n_constr << std::endl;
        sdp_mss = sdp_branch_and_bound(k, Ws, part_constraints, ub_sol);
        std::cout << std::endl << "*********************************************************************" << std::endl;
        std::cout  << std::endl << "UB MSS: " << sdp_mss << std::endl;
        
        if ((ub_mss - sdp_mss) / ub_mss >= 0.0001 or (it == 0)) {
            ub_mss = sdp_mss;
            best_sol = ub_sol;
            save_to_file(ub_sol, result_path, "it" + std::to_string(it));
        }
        if (part_mss < lb_mss)
            improvement = false;
        
        std::cout << std::endl << std::endl << "--------------------------------------------------------------------";
        std::cout << std::endl << "It " << it << " GAP UB-LB " << round((ub_mss - lb_mss) / ub_mss * 100) << "%" << std::endl;
        std::cout << "--------------------------------------------------------------------" << std::endl;
        
        it++;
        
    }
    
    std::cout << std::endl << "*********************************************************************" << std::endl;
    std::cout  << std::endl << "Best UB MSS: " << ub_mss << std::endl;
    std::cout << std::endl << "*********************************************************************" << std::endl;
    
    ub_mss = solve_with_ray(Ws, best_sol, k, result_path);
    
    return std::make_pair(lb_mss, ub_mss);
    
}