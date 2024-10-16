//
// Created by moden on 27/05/2024.
//

#include <armadillo>
#include "Kmeans.h"
#include "Kmeans_max.h"
#include "kmeans_util.h"
#include "matlab_util.h"
#include "mount_model.h"
#include "antic_model.h"
#include "cluster_model.h"
#include "glover_model.h"
#include "sdp_branch_and_bound.h"
#include "ThreadPoolPartition.h"
#include "ThreadPoolAnti.h"
#include "ac_heuristic.h"

void save_to_file(arma::mat &X, std::string name){

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

// generate must link constraints on partition sol
int generate_part_constraints(std::map<int, arma::mat> &sol_map, UserConstraints &constraints) {

    int nc = 0;

    for (int h = 0; h < p; h++) {
    	arma::mat sol = sol_map[h];
    	arma::vec point_id = sol.col(0);
    	arma::vec cls_id = sol.col(sol.n_cols-1);
    	int np = sol.n_rows;

    	for (int c = 0; c < k; c++) {
        	std::list<int> cls_points = {};
        	for (int i = 0; i < np; i++) {
            	if (cls_id(i) == c) {
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

*/
// compute ub
double compute_ub(arma::mat &Ws, arma::mat &sol, std::map<int, arma::mat> &sol_map) {

    std::cout << std::endl << "\n\nGenerate new UB";
    std::cout << std::endl << "*********************************************************************" << std::endl;
    UserConstraints part_constraints;
    int n_constr = generate_part_constraints(sol_map, part_constraints);
    std::cout << std::endl << "Must Link constraints: " << n_constr << std::endl;
    log_file << "\n\nGenerate new UB (Must Link constraints " << n_constr << ")\n";
    double ub_mss;
    sdp_branch_and_bound(k, Ws, ub_mss, part_constraints, sol);

    return ub_mss;
}

arma::mat save_ub(arma::mat &data, arma::mat &sol) {

    arma::mat cls(sol.n_rows,1);
    arma::mat id(sol.n_rows,2);
    for (int i = 0; i < data.n_rows; i++) {
        for (int c = 0; c < k; c++) {
        	if (sol(i,c)==1) {
        		cls(i) = c;
        		id(i,0) = 0;
        		id(i,1) = i+1;
        	}
        }
    }
    arma::mat ub_sol = std::move(arma::join_horiz(id, data));
    ub_sol = std::move(arma::join_horiz(ub_sol, cls));

    return ub_sol;

}


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
        for (int h = 0; h < p; h++) {
            if (h < nc % p)
                pp[h] = floor(nc/p) + 1;
            else
                pp[h] = floor(nc/p);
            part_points[h].resize(pp[h]);
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
                }
                part_points[h][t] = i;
            }
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
                            swap_obj += part_dist[h3];

                    double dist_h1 = part_dist[h1];
                    double dist_h2 = part_dist[h2];

                    for (int ids = 0; ids < pp[h1]; ids++) {
                        int s = part_points[h1][ids];
                        if (s != i)
                            dist_h1 += w_diversity*(all_dist[j][s] - all_dist[i][s]);
                        else
                            idh1 = ids;
                    }

                    for (int ids = 0; ids < pp[h2]; ids++) {
                        int s = part_points[h2][ids];
                        if (s != j)
                            dist_h2 += w_diversity*(all_dist[i][s] - all_dist[j][s]);
                        else
                            idh2 = ids;
                    }

                    swap_obj += dist_h1 + dist_h2;

                    // Update `best` if objective was improved
                    if (swap_obj > best_obj) {
                        best_obj = swap_obj;
                        best_h1 = dist_h1;
                        best_h2 = dist_h2;
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

void heuristic(arma::mat &data, HResult &results) {

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

	// create vector of distances from cluster centroid and create cluster assignments
	std::vector<std::vector<int>> cls(k);
    for (int c = 0; c < k; c++)
		cls[c].reserve(n);

	std::vector<std::vector<double>> all_data(n, std::vector<double>(d));
	std::vector<std::vector<double>> centroid(k, std::vector<double>(d));
    for (int c = 0; c < k; c++) {
    	int nc = 0;
		for (int i = 0; i < n; ++i) {
    		if (results.heu_sol(i,c) == 1) {
				for (int l = 0; l < d; l++) {
					all_data[i][l] = data(i,l);
					centroid[c][l] += data(i,l);
				}
    			cls[c].push_back(i);
    			nc++;
    		}
    	}
		for (int l = 0; l < d; l++)
			centroid[c][l] /= nc;


    	for (int c = 0; c < k; c++)
			for (int l = 0; l < d; l++) {
    			centroid[c][l] = 0;
				for (int i = 0; i < n; ++i)
					centroid[c][l] += all_data[i][l]/n;
			}



    }

	auto start_time_h = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::vector<int>>> sol_cls(k);
    for (int c = 0; c < k; ++c)
        sol_cls[c] = std::vector<std::vector<int>>(p);

    auto *shared_data_anti = new SharedDataAnti();

    shared_data_anti->threadStates.reserve(n_threads_anti);
    for (int i = 0; i < n_threads_anti; i++)
        shared_data_anti->threadStates.push_back(false);
    shared_data_anti->all_data = all_data;
    shared_data_anti->all_dist = all_dist;
    shared_data_anti->sol_cls = sol_cls;

	// create pool of job (1 for each cluster)
    for (int c = 0; c < k; c++) {
        auto *job = new AntiJob();
        job->cls_id = c;
        job->cls_points = cls[c];
        job->center = centroid[c];
    	shared_data_anti->queue.push_back(job);
    }

	log_file << "\n\nAnticlustering Problem Gaps\n";
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
    results.anti_obj = 0;
    for (auto &obj : shared_data_anti->dist_cls)
        results.anti_obj += obj;

    sol_cls = shared_data_anti->sol_cls;

    std::cout << std::endl << std::endl << "Heuristic total dist " << std::fixed << results.anti_obj << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl << std::endl;

    a_pool.quitPool();

    // free memory
	delete (shared_data_anti);

	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();

	auto start_time_m = std::chrono::high_resolution_clock::now();

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

//    	delete model;
    	delete env;

    } catch (GRBException &e) {
    	std::cout << "Error code = " << e.getErrorCode() << std::endl;
    	std::cout << e.getMessage() << std::endl;
    }

	// save mount time
	auto end_time_m = std::chrono::high_resolution_clock::now();
	results.m_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

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

    	save_to_file(data_part, "PART_" + std::to_string(h));
    }

    ThreadPoolPartition p_pool(shared_data_part, n_threads_part);
	log_file << "\n";

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
    for (auto &lb_bound : shared_data_part->lb_part)
        results.lb_mss += lb_bound;
    results.ub_mss = 0;
    for (auto &ub_bound : shared_data_part->ub_part)
        results.ub_mss += ub_bound;

	std::map<int, arma::mat> sol_map;
    for (int h = 0; h < p; h++) {
    	arma::vec arma_sol = arma::conv_to<arma::vec>::from(sol[h]);
    	sol_map[h] = std::move(arma::join_horiz(arma_sol, shared_data_part->sol_part[h]));
    }

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
	results.lb_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

    // create upper bound
    arma::mat sol_ub;
    double new_ub = compute_ub(data, sol_ub, sol_map);
    arma::mat sdp_sol = save_ub(data, sol_ub);
	for (int h = 0; h < p; h++)
		for (int row : sol_map[h].col(0))
			sdp_sol(row,0) = h;


	std::cout << std::endl << "*********************************************************************" << std::endl;
	std::cout  << std::endl << "UB MSS: " << new_ub << std::endl;

    log_file << "\nSum of WSS\n";
    log_file << "LB: " << results.lb_mss;
    log_file << "\nUB: " << results.ub_mss;

    log_file << "\n\nStart sol: " << results.heu_mss;
    log_file << "\nNew UB: " << new_ub << "\n\n\n";

	results.heu_mss = new_ub;
	results.heu_sol = sol_ub;
	save_to_file(sdp_sol, "UB");

}

void heuristic_no_sol(arma::mat &data, HResult &results) {

	std::cout << std::endl << "Running heuristics anticluster..";

	std::vector<std::vector<double>> all_data(n, std::vector<double>(d));
	for (int i = 0; i < n; ++i)
		for (int l = 0; l < d; l++)
			all_data[i][l] = data(i,l);

	auto start_time_h = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int>> sol(p);

    	try {
    		GRBEnv *env = new GRBEnv();
    		antic_model *model = new antic_gurobi_model(env, n, p, d);

	    	model->add_point_constraints();
    		model->add_part_constraints();
		    model->add_dev_constraints(all_data);

	    	model->optimize();
    		model->get_x_solution(sol);

	//    	delete model;
    		delete env;

	    } catch (GRBException &e) {
    		std::cout << "Error code = " << e.getErrorCode() << std::endl;
    		std::cout << e.getMessage() << std::endl;
    	}


	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();

	auto start_time_m = std::chrono::high_resolution_clock::now();
	auto end_time_m = std::chrono::high_resolution_clock::now();
	results.m_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

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

    	save_to_file(data_part, "PART_" + std::to_string(h));
    }

    ThreadPoolPartition p_pool(shared_data_part, n_threads_part);
	log_file << "\n";

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
    for (auto &lb_bound : shared_data_part->lb_part)
        results.lb_mss += lb_bound;
    results.ub_mss = 0;
    for (auto &ub_bound : shared_data_part->ub_part)
        results.ub_mss += ub_bound;

	std::map<int, arma::mat> sol_map;
    for (int h = 0; h < p; h++) {
    	arma::vec arma_sol = arma::conv_to<arma::vec>::from(sol[h]);
    	sol_map[h] = std::move(arma::join_horiz(arma_sol, shared_data_part->sol_part[h]));
    }

    p_pool.quitPool();

    // free memory
    delete (shared_data_part);

	auto end_time_lb = std::chrono::high_resolution_clock::now();
	results.lb_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

    // create upper bound
    arma::mat sol_ub;
    double new_ub = compute_ub(data, sol_ub, sol_map);
    arma::mat sdp_sol = save_ub(data, sol_ub);
	for (int h = 0; h < p; h++)
		for (int row : sol_map[h].col(0))
			sdp_sol(row,0) = h;


	std::cout << std::endl << "*********************************************************************" << std::endl;
	std::cout  << std::endl << "UB MSS: " << new_ub << std::endl;

    log_file << "\nSum of WSS\n";
    log_file << "LB: " << results.lb_mss;
    log_file << "\nUB: " << results.ub_mss;

    log_file << "\n\nStart sol: " << results.heu_mss;
    log_file << "\nNew UB: " << new_ub << "\n\n\n";

	results.heu_mss = new_ub;
	results.heu_sol = sol_ub;
	save_to_file(sdp_sol, "UB");

}


void heuristic_kmeans(arma::mat &data, HResult &results) {

	std::cout << std::endl << "Running heuristics anticluster with k-means max..";

	// create matrix of all distances
    std::vector<std::vector<double>> all_dist(n, std::vector<double>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = i+1; j < n; ++j) {
			double dist = std::pow(arma::norm(data.row(i).t() - data.row(j).t(), 2),2);
			all_dist[i][j] = dist;
			all_dist[j][i] = dist;
		}
	}


	auto start_time_h = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::vector<int>>> sol_cls(k);

	for (int c = 0; c < k; c++) {

	 	arma::mat cls_data = arma::zeros(n,d);
	 	std::vector<int> cls_points(n);
    	int nc = 0;
		for (int i = 0; i < n; ++i) {
    		if (results.heu_sol(i,c) == 1) {
				cls_data.row(nc) = data.row(i);
    			cls_points[nc] = i;
    			nc++;
    		}
    	}
    	cls_data = cls_data.rows(0, nc-1);

    	std::map<int, std::set<int>> ml_map = {};
    	std::vector <std::pair<int, int>> local_cl = {};
    	std::vector <std::pair<int, int>> global_ml = {};
    	std::vector <std::pair<int, int>> global_cl = {};

		arma::mat cls_sol(nc,p);
    	Kmeans_max kmeans_max(cls_data, p, ml_map, local_cl, global_ml, global_cl, 1);
    	kmeans_max.start(kmeans_max_it, kmeans_start, kmeans_permut);
    	cls_sol = kmeans_max.getAssignments();

        sol_cls[c] = std::vector<std::vector<int>>(p);
    	for (int h = 0; h < p; h++) {
			int points = floor(nc/p);
			if (h < nc % p)
				points += 1;
			sol_cls[c][h].reserve(points);
			for (int i = 0; i < nc; i++) {
        		if (cls_sol(i,h) == 1)
        			sol_cls[c][h].push_back(cls_points[i]);
        	}
		}

    }

	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();

	auto start_time_m = std::chrono::high_resolution_clock::now();

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

//    	delete model;
    	delete env;

    } catch (GRBException &e) {
    	std::cout << "Error code = " << e.getErrorCode() << std::endl;
    	std::cout << e.getMessage() << std::endl;
    }

	// save mount time
	auto end_time_m = std::chrono::high_resolution_clock::now();
	results.m_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

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

    	save_to_file(data_part, "PART_" + std::to_string(h));
    }

    ThreadPoolPartition p_pool(shared_data_part, n_threads_part);
	log_file << "\n";

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
    for (auto &lb_bound : shared_data_part->lb_part)
        results.lb_mss += lb_bound;
    results.ub_mss = 0;
    for (auto &ub_bound : shared_data_part->ub_part)
        results.ub_mss += ub_bound;

	std::map<int, arma::mat> sol_map;
    for (int h = 0; h < p; h++) {
    	arma::vec arma_sol = arma::conv_to<arma::vec>::from(sol[h]);
    	sol_map[h] = std::move(arma::join_horiz(arma_sol, shared_data_part->sol_part[h]));
    }

    p_pool.quitPool();

    // free memory
    delete (shared_data_part);

	auto end_time_lb = std::chrono::high_resolution_clock::now();
	results.lb_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

    // create upper bound
    arma::mat sol_ub;
    double new_ub = compute_ub(data, sol_ub, sol_map);
    arma::mat sdp_sol = save_ub(data, sol_ub);
	for (int h = 0; h < p; h++)
		for (int row : sol_map[h].col(0))
			sdp_sol(row,0) = h;


	std::cout << std::endl << "*********************************************************************" << std::endl;
	std::cout  << std::endl << "UB MSS: " << new_ub << std::endl;

    log_file << "\nSum of WSS\n";
    log_file << "LB: " << results.lb_mss;
    log_file << "\nUB: " << results.ub_mss;

    log_file << "\n\nStart sol: " << results.heu_mss;
    log_file << "\nNew UB: " << new_ub << "\n\n\n";

	results.heu_mss = new_ub;
	results.heu_sol = sol_ub;
	save_to_file(sdp_sol, "UB");

}

void heuristic_new(arma::mat &data, HResult &results) {

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

	int size_y = 0;
	std::vector<std::vector<int>> cls(k);
	for (int c = 0; c < k; c++) {
	 	cls[c].reserve(n);
    	int nc = 0;
		for (int i = 0; i < n; ++i) {
    		if (results.heu_sol(i,c) == 1) {
    			cls[c].push_back(i);
    			nc++;
    		}
    	}
    	size_y += nc*(nc -1)/2;
    }

	auto start_time_h = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int>> sol(p);

    	try {
    		GRBEnv *env = new GRBEnv();
    		glover_model *model = new glover_gurobi_model(env, n, p, k, all_dist, cls);

	    	model->add_point_constraints();
    		model->add_part_constraints();
    		model->add_bound_constraints();

	    	model->optimize();
    		model->get_x_solution(sol);

	//    	delete model;
    		delete env;

	    } catch (GRBException &e) {
    		std::cout << "Error code = " << e.getErrorCode() << std::endl;
    		std::cout << e.getMessage() << std::endl;
    	}


	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();

	auto start_time_m = std::chrono::high_resolution_clock::now();
	auto end_time_m = std::chrono::high_resolution_clock::now();
	results.m_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

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

    	save_to_file(data_part, "PART_" + std::to_string(h));
    }

    ThreadPoolPartition p_pool(shared_data_part, n_threads_part);
	log_file << "\n";

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
    for (auto &lb_bound : shared_data_part->lb_part)
        results.lb_mss += lb_bound;
    results.ub_mss = 0;
    for (auto &ub_bound : shared_data_part->ub_part)
        results.ub_mss += ub_bound;

	std::map<int, arma::mat> sol_map;
    for (int h = 0; h < p; h++) {
    	arma::vec arma_sol = arma::conv_to<arma::vec>::from(sol[h]);
    	sol_map[h] = std::move(arma::join_horiz(arma_sol, shared_data_part->sol_part[h]));
    }

    p_pool.quitPool();

    // free memory
    delete (shared_data_part);

	auto end_time_lb = std::chrono::high_resolution_clock::now();
	results.lb_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_lb - start_time_lb).count();

    // create upper bound
    arma::mat sol_ub;
    double new_ub = compute_ub(data, sol_ub, sol_map);
    arma::mat sdp_sol = save_ub(data, sol_ub);
	for (int h = 0; h < p; h++)
		for (int row : sol_map[h].col(0))
			sdp_sol(row,0) = h;


	std::cout << std::endl << "*********************************************************************" << std::endl;
	std::cout  << std::endl << "UB MSS: " << new_ub << std::endl;

    log_file << "\nSum of WSS\n";
    log_file << "LB: " << results.lb_mss;
    log_file << "\nUB: " << results.ub_mss;

    log_file << "\n\nStart sol: " << results.heu_mss;
    log_file << "\nNew UB: " << new_ub << "\n\n\n";

	results.heu_mss = new_ub;
	results.heu_sol = sol_ub;
	save_to_file(sdp_sol, "UB");

}