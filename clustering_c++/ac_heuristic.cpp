//
// Created by moden on 27/05/2024.
//
#include "Kmeans.h"
#include "kmeans_util.h"
#include "matlab_util.h"
#include "mount_model.h"
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
std::map<int, arma::mat> read_part_data(int n, int d, int k, int p, arma::mat data) {

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

// compute lb
double compute_lb(std::map<int, arma::mat> &sol_map, int k) {

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
        std::cout << "Part " << part.first << std::endl;
		arma::mat centr(k, d);
		arma::vec count(k);
        for (int i = 0; i < np; i++)
            for (int c = 0; c < k; c++)
                if (sol(i,c)==1) {
                    cls(i)= c+1;
                    centr.row(c) += data.row(i);
                    count(c)++;
                }
        for (int c = 0; c < k; c++)
        	std::cout << "cluster " << c << "centroid: " << centr.row(c)/count(c) << std::endl;
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

HResult heuristic(arma::mat data, int p, int k) {

	HResult results;
    double lb_mss;
    double ub_mss;
    arma::mat sdp_sol;
	UserConstraints constraints;
	std::map<int, arma::mat> sol_map;

    // generating lb
	int n = data.n_rows;
	int d = data.n_cols;

	std::cout << "Running heuristics per cluster.." << std::endl;

	// create matrix of all distances
	double max_d = 0;
    results.h_obj = 0;
    std::vector<std::vector<double>> all_dist(n, std::vector<double>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = i+1; j < n; ++j) {
			double dist = std::pow(arma::norm(data.row(i).t() - data.row(j).t(), 2),2);
			all_dist[i][j] = dist;
			all_dist[j][i] = dist;
			if (dist > max_d)
				max_d = all_dist[j][i];
		}
	}

	// create map of cluster assignment
	std::unordered_map<int, std::vector<int>> cls;
	cls.reserve(k);
    for (int c = 0; c < k; c++)
		cls[c].reserve(n);

    std::vector<int> cp(k);
	for (int i = 0; i < n; ++i)
    	for (int c = 0; c < k; c++)
    		if (init_sol(i,c) == 1) {
    			cls[c].push_back(i);
    			cp[c]++;
    		}

	for (int c = 0; c < k; c++)
		cls[c].resize(cp[c]);

	int num_update = 0;
	std::map<int, std::map<int, arma::mat> > sol_cls;

	auto start_time_h = std::chrono::high_resolution_clock::now();

    for (int c = 0; c < k; c++) {

    	std::cout << "Cluster " << c << std::endl;
    	std::vector<int> cls_points = cls[c];
    	int nc = cp[c];

        /* Start main iteration loop for exchange procedure */
    	std::unordered_map<int, std::vector<int>> best_part_map;
        double best_dist = 0;

        for (int l = 0; l < 100; l++) {

			// create partition vector
        	double dist = 0;
        	std::unordered_map<int, std::vector<int>> part_map(p);
        	std::vector<double> part_dist(p);
        	std::vector<int> part(n);
        	std::vector<int> pp(p);
        	std::vector<double> min_dist(p);
        	for (int h = 0; h < p; h++) {
        		if (h < nc % p)
        			pp[h] = floor(nc/p) + 1;
        		else
        			pp[h] = floor(nc/p);
        		part_map[h].resize(pp[h]);
        	}

        	// assign random point to partitions and compute dist of current partition
        	std::vector<int> cls_copy = cls_points;
        	for (int h = 0; h < p && !cls_copy.empty(); ++h) {
        		for (int t = 0; t < pp[h]; ++t) {
        			int idx = rand() % (cls_copy.size());
        			int i = cls_copy[idx];
        			cls_copy.erase(cls_copy.begin() + idx);

        			part[i] = h;
        			for (const int& j : part_map[h]) {
        				dist += all_dist[i][j];
        				part_dist[h] += all_dist[i][j];
        				if (min_dist[h] > all_dist[i][j])
        					min_dist[h] = all_dist[i][j];
        			}
        			part_map[h][t] = i;
        		}
        		dist += min_dist[h];
        	}

        	if (l == 0) {
				best_part_map = part_map;
				best_dist = dist;
        	}

        	// Iterate through cluster data points
    		for (int id1 = 0; id1 < nc - 1; ++id1) {

    			int i = cls_points[id1];
            	double best_obj = dist;
            	int h1 = part[i];
            	int idh1 = -1;

	            // Initialize `best` variable for the i'th item
    	        double best_h1 = 0;
        		double best_h2 = 0;
        		double best_min_h1 = min_dist[h1];
        		double best_min_h2 = max_d;
            	std::pair<int, int> best_swap(NULL,NULL);

	    		// Iterate through the exchange partners
    			for (int id2 = id1+1; id2 < nc; ++id2) {
    				int j = cls_points[id2];

    				int h2 = part[j];
            		int idh2 = -1;

        	        if (h2 != h1) {

        	        	double swap_obj = 0;

                	    for (int h3= 0; h3 < p; h3++)
                    		if (h3 != h1 and h3!=h2)
                    			swap_obj += part_dist[h3] + min_dist[h3];

        	        	double dist_h1 = part_dist[h1];
        	        	double min_dist_h1 = min_dist[h1];
        	        	for (int ids = 0; ids < pp[h1]; ids++) {
        	        		int s = part_map[h1][ids];
            	    		if (s != i) {
                				dist_h1 += all_dist[j][s] - all_dist[i][s];
            	    			if (min_dist_h1 > all_dist[j][s])
            	    				min_dist_h1 = all_dist[j][s];
                			}
                			else
                				idh1 = ids;
             		   	}

        	        	double dist_h2 = part_dist[h2];
        	        	double min_dist_h2 = min_dist[h2];
        	        	for (int ids = 0; ids < pp[h2]; ids++) {
        	        		int s = part_map[h2][ids];
            	    		if (s != j) {
            	    			dist_h2 += all_dist[i][s] - all_dist[j][s];
            	    			if (min_dist_h2 > all_dist[i][s])
            	    				min_dist_h2 = all_dist[i][s];
                			}
            	    		else
            	    			idh2 = ids;
                		}

                		swap_obj += dist_h1 + dist_h2 + min_dist_h1 + min_dist_h2;

	                    // Update `best` if objective was improved
    	                if (swap_obj > best_obj) {
        	                best_obj = swap_obj;
            	            best_h1 = dist_h1;
                	    	best_h2 = dist_h2;
                	    	best_min_h1 = min_dist_h1;
                	    	best_min_h2 = min_dist_h2;
                    	    best_swap = std::pair<int, int>(id2, idh2);
                    	}
                	}
            	}

            	// Only if objective is improved: Do the swap
            	if ( best_obj > dist) {
	                dist = best_obj;
    	            int id2 = best_swap.first;
    	            int idh2 = best_swap.second;
    	            int j = cls_points[id2];
        	        int h2 = part[j];
            	    part_map[h1][idh1] = j;
            	    part_map[h2][idh2] = i;
        	        part[i] = h2;
            	    part[j] = h1;
                	part_dist[h1] = best_h1;
                	part_dist[h2] = best_h2;
                	min_dist[h1] = best_min_h1;
                	min_dist[h2] = best_min_h2;
                	best_swap = std::pair<int, int>(NULL, NULL);
           			num_update++;
           		}
        	}

	        if (dist > best_dist) {
    	    	best_dist = dist;
        		best_part_map = part_map;
       		}

    	}


    	results.h_obj += best_dist;

    	// update sol map
    	for (int h=0; h < p; ++h) {
    		sol_cls[c][h] = arma::mat(nc,d+1);
    		int np = 0;
    		for (int i : best_part_map[h]) {
    			sol_cls[c][h](np,0) = i+1;
    			sol_cls[c][h].row(np).subvec(1,d) = data.row(i);
    			np++;
    		}
    		sol_cls[c][h] = sol_cls[c][h].submat(0, 0, np - 1, d);
    	}


    }

	std::cout << std::endl << std::endl << "Heuristic total dist " << results.h_obj << std::endl;
	std::cout << "Num update " << num_update << std::endl << std::endl;

    // mount cluster partitions
    try {
    	GRBEnv *env = new GRBEnv();
    	mount_model *model = new mount_gurobi_model(env, n, p, k, k*p*(k-1)*p/2, compute_distances(data), sol_cls);

    	model->add_point_constraints();
    	model->add_cls_constraints();
    	model->add_edge_constraints();

    	model->optimize();

    	std::map<int, arma::vec> sol = model->get_x_solution();

    	// update sol map
    	for (int t=0; t < p; ++t) {
    		for (int c = 0; c < k; c++) {
    			if (c==0)
    				sol_map[t] = sol_cls[c][sol[t](c)];
    			else
    				sol_map[t] = std::move(arma::join_vert(sol_map[t], sol_cls[c][sol[t](c)]));
    		}
    	}



    	delete model;
    	delete env;

    } catch (GRBException &e) {
    	std::cout << "Error code = " << e.getErrorCode() << std::endl;
    	std::cout << e.getMessage() << std::endl;
    }


	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time = std::chrono::duration_cast<std::chrono::minutes>(end_time_h - start_time_h).count();

	// create lower bound
	auto start_time_lb = std::chrono::high_resolution_clock::now();
    results.lb_mss = compute_lb(sol_map, k);
    sdp_sol = save_lb(sol_map, p);
	save_to_file(sdp_sol, "LB_method_" + std::string(1,part_m));
	auto end_time_lb = std::chrono::high_resolution_clock::now();
	results.lb_time = std::chrono::duration_cast<std::chrono::minutes>(end_time_lb - start_time_lb).count();

    // create upper bound
	auto start_time_ub = std::chrono::high_resolution_clock::now();
    results.ub_mss = compute_ub(data, sdp_sol, sol_map, k, p);
    sdp_sol = save_ub(data	, sdp_sol);
	save_to_file(sdp_sol, "UB_method_" + std::string(1,part_m));
	auto end_time_ub = std::chrono::high_resolution_clock::now();
	results.ub_time = std::chrono::duration_cast<std::chrono::minutes>(end_time_ub - start_time_ub).count();

	results.all_time = results.h_time + results.lb_time + results.ub_time;


    return results;

}