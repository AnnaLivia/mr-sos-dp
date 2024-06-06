//
// Created by moden on 27/05/2024.
//

#include <armadillo>
#include "Kmeans.h"
#include "kmeans_util.h"
#include "matlab_util.h"
#include "mount_model.h"
#include "sdp_branch_and_bound.h"
#include "ThreadPoolPartition.h"
#include "ThreadPoolAnti.h"
#include "ac_heuristic.h"

void save_to_file(arma::mat &X, std::string name){

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

// compute mss for a given solution
double compute_mss(arma::mat &data, arma::mat &sol) {

    int data_n = data.n_rows;
    int data_d = data.n_cols;
    int data_k = sol.n_cols;

    if (data_n != sol.n_rows) {
        std::printf("compute_mss() - ERROR: inconsistent data and sol!\n");
        exit(EXIT_FAILURE);
    }

    arma::mat assignment_mat = arma::zeros(data_n, data_k);
    arma::vec count = arma::zeros(data_k);
    arma::mat centroids = arma::zeros(data_k, data_d);
    for (int i = 0; i < data_n; i++) {
        for (int j = 0; j < data_k; j++) {
            if (sol(i,j) == 1) {
                assignment_mat(i, j) = 1;
                ++count(j);
                centroids.row(j) += data.row(i);
            }
        }
    }

    // compute clusters' centroids
    for (int c = 0; c < data_k; c++) {
        // empty cluster
        if (count(c) == 0) {
            std::printf("read_data(): cluster %d is empty!\n", c);
            return false;
        }
        centroids.row(c) = centroids.row(c) / count(c);
    }

    arma::mat m = data - assignment_mat * centroids;

    return arma::dot(m.as_col(), m.as_col());
}

// read lb data
std::map<int, arma::mat> read_part_data(arma::mat &data) {

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
int generate_part_constraints(std::map<int, arma::mat> &sol_map, UserConstraints &constraints) {

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

/*
// compute lb
double compute_lb(std::map<int, arma::mat> &sol_map) {

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
    log_file << "\n\nMerge LB MSS: " << lb_mss << "\n\n\n";

    return lb_mss;
}

// compute ub
double compute_ub(arma::mat &Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map) {

    std::cout << std::endl << "Generating UB";
    std::cout << std::endl << "*********************************************************************" << std::endl;
    UserConstraints part_constraints;
    int n_constr = generate_part_constraints(sol_map, part_constraints);
    std::cout << std::endl << "Added constraints: " << n_constr << std::endl;
    log_file << "Generating UB (added constraints " << n_constr << ")\n";
    double ub_mss = sdp_branch_and_bound(k, Ws, part_constraints, sol);
    std::cout << std::endl << "*********************************************************************" << std::endl;
    std::cout  << std::endl << "UB MSS: " << ub_mss << std::endl;

    return ub_mss;
}

arma::mat save_ub(arma::mat &data, arma::mat &sol) {

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
*/


std::pair<double, std::vector<std::vector<int>>> compute_anti_cls(std::vector<int> &cls_points, std::vector<std::vector<double>> &all_dist) {

    int nc = (int) cls_points.size();

    /* Start main iteration loop for exchange procedure */
    std::vector<std::vector<int>> best_part_points;
    double best_dist = 0;

    for (int l = 0; l < num_rep; l++) {

        // create partition vector
        double dist = 0;
        std::vector<std::vector<int>> part_points(p);
        std::vector<double> part_dist(p);
        std::vector<int> part(n);
        std::vector<int> pp(p);
        std::vector<double> min_dist(p);
        for (int h = 0; h < p; h++) {
            if (h < nc % p)
                pp[h] = floor(nc/p) + 1;
            else
                pp[h] = floor(nc/p);
            part_points[h].resize(pp[h]);
            min_dist[h] = std::numeric_limits<double>::infinity();
        }

        // assign random point to partitions and compute dist of current partition and min dist
        std::vector<int> cls_copy = cls_points;
        for (int h = 0; h < p && !cls_copy.empty(); ++h) {
            for (int t = 0; t < pp[h]; ++t) {
                int idx = rand() % (cls_copy.size());
                int i = cls_copy[idx];
                cls_copy.erase(cls_copy.begin() + idx);

                part[i] = h;
                for (const int& j : part_points[h]) {
                    dist += w_diversity*all_dist[i][j];
                    part_dist[h] += all_dist[i][j];
                    if (min_dist[h] > all_dist[i][j])
                        min_dist[h] = all_dist[i][j];
                }
                part_points[h][t] = i;
            }
            dist += w_dispersion*min_dist[h];
        }

        if (l == 0) {
            best_part_points = part_points;
            best_dist = dist;
        }

        // Iterate through cluster data points
        for (int id1 = 0; id1 < nc - 1; ++id1) {

            int i = cls_points[id1];
            double best_obj = dist;
            int h1 = part[i];
            int idh1 = -1;

            // Initialize `best` variable for the i-th item
            double best_h1 = 0;
            double best_h2 = 0;
            double best_min_h1 = min_dist[h1];
            double best_min_h2;
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
                    double dist_h2 = part_dist[h2];
                    double min_dist_h2 = min_dist[h2];

                    for (int s : part_points[h1]) {
                    	if (all_dist[i][s] == min_dist_h1) {
                    		// compute new min dist for partition h1
                    		min_dist_h1 = std::numeric_limits<double>::infinity();
                    		for (int s1 : part_points[h1])
                    			for (int s2 : part_points[h1])
                            		if (s1!=s2 and min_dist_h1 > all_dist[s1][s2])
                                		min_dist_h1 = all_dist[s1][s2];
                            break;
                    	}
                    }

                    for (int s : part_points[h2]) {
                    	if (all_dist[j][s] == min_dist_h2) {
                    		// compute new min dist for partition h2
                    		min_dist_h2 = std::numeric_limits<double>::infinity();
                    		for (int s1 : part_points[h2])
                    			for (int s2 : part_points[h2])
                            		if (s1!=s2 and min_dist_h2 > all_dist[s1][s2])
                                		min_dist_h2 = all_dist[s1][s2];
                            break;
                    	}
                    }

                    for (int ids = 0; ids < pp[h1]; ids++) {
                        int s = part_points[h1][ids];
                        if (s != i) {
                            dist_h1 += w_diversity*(all_dist[j][s] - all_dist[i][s]);
                            if (min_dist_h1 > all_dist[j][s])
                                min_dist_h1 = all_dist[j][s];
                        }
                        else
                            idh1 = ids;
                    }

                    for (int ids = 0; ids < pp[h2]; ids++) {
                        int s = part_points[h2][ids];
                        if (s != j) {
                            dist_h2 += w_diversity*(all_dist[i][s] - all_dist[j][s]);
                            if (min_dist_h2 > all_dist[i][s])
                                min_dist_h2 = all_dist[i][s];
                        }
                        else
                            idh2 = ids;
                    }

                    swap_obj += dist_h1 + dist_h2 + w_dispersion*(min_dist_h1 + min_dist_h2);

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
            if (best_obj > dist) {
                dist = best_obj;
                int id2 = best_swap.first;
                int idh2 = best_swap.second;
                int j = cls_points[id2];
                int h2 = part[j];
                part_points[h1][idh1] = j;
                part_points[h2][idh2] = i;
                part[i] = h2;
                part[j] = h1;
                part_dist[h1] = best_h1;
                part_dist[h2] = best_h2;
                min_dist[h1] = best_min_h1;
                min_dist[h2] = best_min_h2;
                best_swap = std::pair<int, int>(NULL, NULL);
            }
        }

        if (dist > best_dist) {
            best_dist = dist;
            best_part_points = part_points;
        }

    }

    return std::make_pair(best_dist, best_part_points);

}

HResult heuristic(arma::mat &data) {

	HResult results;
	arma::mat dis = compute_distances(data);
	std::cout << "max dist " << arma::max(arma::max(dis));
	std::cout << std::endl << "Running heuristics anticluster..";

	// create matrix of all distances
    std::vector<std::vector<double>> all_dist(n, std::vector<double>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = i+1; j < n; ++j) {
			double dist = std::pow(arma::norm(data.row(i).t() - data.row(j).t(), 2),2);
			all_dist[i][j] = dist;
			all_dist[j][i] = dist;
		}
	}

	// create cluster assignments
	std::vector<std::vector<int>> cls(k);
    for (int c = 0; c < k; c++)
		cls[c].reserve(n);

	for (int i = 0; i < n; ++i)
    	for (int c = 0; c < k; c++)
    		if (init_sol(i,c) == 1)
    			cls[c].push_back(i);


	auto start_time_h = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::vector<int>>> sol_cls(k);
    for (int c = 0; c < k; ++c)
        sol_cls[c] = std::vector<std::vector<int>>(p);

    auto *shared_data_anti = new SharedDataAnti();

    shared_data_anti->threadStates.reserve(n_threads_anti);
    for (int i = 0; i < n_threads_anti; i++)
        shared_data_anti->threadStates.push_back(false);
    shared_data_anti->all_dist = all_dist;
    shared_data_anti->sol_cls = sol_cls;

	// create pool of job (1 for each cluster)
    for (int c = 0; c < k; c++) {
        auto *job = new AntiJob();
        job->cls_id = c;
        job->cls_points = cls[c];
        shared_data_anti->queue.push_back(job);
    }

    ThreadPoolAnti a_pool(shared_data_anti, n_threads_anti);

    while (true) {

        {
            std::unique_lock<std::mutex> l(shared_data_anti->queueMutex);
            while (is_thread_pool_working(shared_data_anti->threadStates))
                shared_data_anti->mainConditionVariable.wait(l);

            if (shared_data_anti->queue.empty())
                break;
        }

    }

    // collect all the results
    results.h_obj = 0;
    for (auto &obj : shared_data_anti->dist_cls)
        results.h_obj += obj;

    sol_cls = shared_data_anti->sol_cls;

    std::cout << std::endl << std::endl << "Heuristic total dist " << std::fixed << results.h_obj << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl << std::endl;
    a_pool.quitPool();

    // free memory
    delete (shared_data_anti);

    // mount cluster partitions
    std::vector<std::vector<int>> sol(p);

    try {
    	GRBEnv *env = new GRBEnv();
    	mount_model *model = new mount_gurobi_model(env, k*(k-1)*p*p/2, all_dist, sol_cls);

    	model->add_point_constraints();
    	model->add_cls_constraints();
    	model->add_edge_constraints();

    	model->optimize();

    	sol = model->get_x_solution(sol_cls);

    	delete model;
    	delete env;

    } catch (GRBException &e) {
    	std::cout << "Error code = " << e.getErrorCode() << std::endl;
    	std::cout << e.getMessage() << std::endl;
    }

	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();


	// create lower bound
	auto start_time_lb = std::chrono::high_resolution_clock::now();

    auto *shared_data_part = new SharedDataPartition();

    shared_data_part->threadStates.reserve(n_threads_part);
    for (int i = 0; i < n_threads_part; i++)
        shared_data_part->threadStates.push_back(false);

    for (int h = 0; h < p; ++h) {
        auto *job = new PartitionJob();
        job->part_id = h;

        // scale part data if necessary
        arma::mat data_part(sol[h].size(), d);
		for (int i = 0; i < sol[h].size(); i++)
			data_part.row(i) = data.row(sol[h][i]);
        job->part_data = data_part;
        shared_data_part->queue.push_back(job);
    }

    ThreadPoolPartition p_pool(shared_data_part, n_threads_part);

    while (true) {

        {
            std::unique_lock<std::mutex> l(shared_data_part->queueMutex);
            while (is_thread_pool_working(shared_data_part->threadStates)) {
                shared_data_part->mainConditionVariable.wait(l);
            }

            if (shared_data_part->queue.empty())
                break;
        }

    }

    // collect all the results
    results.lb_mss = 0;
    for (auto &bound : shared_data_part->lb_part)
        results.lb_mss += bound;

    log_file << "\n\nMerge LB MSS: " << results.lb_mss << "\n\n\n";

	/*
	// compute lb sol by merging partitions sol
    arma::mat part;
    arma::mat sol;
    for (int h = 0; h < p; ++h) {
        if (h == 0) {
            part = arma::ones(shared_data_part->sol_part[h].n_rows,1);
            sol = arma::join_horiz(part, shared_data_part->sol_part[h]);
        }
        part = arma::vec(shared_data_part->sol_part[h].n_rows,1).fill(h+1);
        arma::mat solp = arma::join_horiz(part, shared_data_part->sol_part[h]);
        sol = arma::join_vert(sol, solp);
    }

	save_to_file(sol, "LB_method_" + std::string(1,part_m));
	*/

    p_pool.quitPool();

    // free memory
    delete (shared_data_part);

	auto end_time_lb = std::chrono::high_resolution_clock::now();
	results.lb_time = std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

	/*
    // create upper bound
    results.ub_mss = compute_ub(data, sol, sol_map, k, p);
    sdp_sol = save_ub(data	, sol);
	save_to_file(sdp_sol, "UB_method_" + std::string(1,part_m));
    */

    return results;

}