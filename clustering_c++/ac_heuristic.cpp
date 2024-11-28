//
// Created by moden on 27/05/2024.
//

#include <armadillo>
#include <iomanip>
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

// compute ub
double compute_ub(arma::mat &data, arma::mat &sol, std::map<int, arma::mat> &sol_map) {

    std::cout << std::endl << "\n\nGenerate new UB";
    std::cout << std::endl << "*********************************************************************" << std::endl;
    UserConstraints part_constraints;
    int n_constr = generate_part_constraints(sol_map, part_constraints);
    std::cout << std::endl << "Must Link constraints: " << n_constr << std::endl;
    log_file << "\n\nGenerate new UB (Must Link constraints " << n_constr << ")\n";
    double ub_mss;
    sdp_branch_and_bound(k, data, ub_mss, part_constraints, sol, true);

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


void heuristic_kmeans(arma::mat &data, HResult &results) {

	std::cout << std::endl << "Running heuristics anticluster with k-means esteeme..\n\n";

	// Create an initial partition and save init centroids
    std::vector<std::vector<std::vector<int>>> sol_cls(k);
	arma::mat centroids_heu = arma::zeros(k, d);
	std::random_device rd;
	std::mt19937 gen(rd());
    for (int c = 0; c < k; ++c) {
        sol_cls[c] = std::vector<std::vector<int>>(p);
    	std::vector<int> all_points;
    	all_points.reserve(n);
        for (int i = 0; i < n; ++i)
    		if (results.heu_sol(i,c) == 1) {
    			all_points.push_back(i);
    			centroids_heu.row(c) += data.row(i);
    		}
    	int nc = all_points.size();
    	if (nc == 0)
    		std::printf("read_data(): cluster %d is empty!\n", c);
    	centroids_heu.row(c) /= nc;

    	std::shuffle(all_points.begin(), all_points.end(), gen);
    	for (int h = 0; h < p; ++h) {
			int points = floor(nc/p);
			if (h <  nc % p)
				points += 1;
    		sol_cls[c][h] = std::vector<int>(points);
        	for (int i = 0; i < points; ++i) {
        		sol_cls[c][h][i] = all_points.back();
        		all_points.pop_back();
            }
    	}
    }

	auto start_time_m = std::chrono::high_resolution_clock::now();

	std::vector<std::vector<double>> all_dist(n, std::vector<double>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = i+1; j < n; ++j) {
			double dist = std::pow(arma::norm(data.row(i).t() - data.row(j).t(), 2),2);
			all_dist[i][j] = dist;
			all_dist[j][i] = dist;
		}
	}

    // mount cluster partitions
    std::vector<std::vector<int>> init_sol(p);

    try {
    	GRBEnv *env = new GRBEnv();
    	mount_model *model = new mount_gurobi_model(env, k*(k-1)*p*p/2, all_dist, sol_cls);

    	model->add_point_constraints();
    	model->add_cls_constraints();
    	model->add_edge_constraints();
    	model->optimize();

    	init_sol = model->get_x_solution(sol_cls);

    	delete env;

    } catch (GRBException &e) {
    	std::cout << "Error code = " << e.getErrorCode() << std::endl;
    	std::cout << e.getMessage() << std::endl;
    }

	// save mount time
	auto end_time_m = std::chrono::high_resolution_clock::now();
	results.m_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

    auto start_time_h = std::chrono::high_resolution_clock::now();

	// Create first sol anticluter
    arma::mat antic_sol = arma::zeros(n,p);
	for (int h = 0; h < p; ++h)
    	for (int i : init_sol[h])
    		antic_sol(i, h) = 1;

	//Create first ub sol
	std::vector<std::vector<std::vector<int>>> points(k);
	std::vector<std::vector<int>> sizes(k);
	for (int c = 0; c < k; c++) {
		points[c] = std::vector<std::vector<int>>(p);
		sizes[c] = std::vector<int>(p);
	}
	arma::mat ub_sol = arma::zeros(n,k);
	arma::mat W_hc = create_first_sol(data, antic_sol, centroids_heu, ub_sol, points, sizes);
	double best_W = arma::accu(W_hc);
	double best_GAP = (results.heu_mss - best_W) / results.heu_mss * 100;

	/*// print first ub
	arma::mat sdp_sol1 = save_ub(data, ub_sol);
	for (int i = 0; i < n; i++)
		for (int h = 0; h < p; h++)
			if (antic_sol(i, h) == 1)
				sdp_sol1(i,0) = h;
	save_to_file(sdp_sol1, "first_UB");
	*/

	log_file << "\n---------------------------------------------------------------------\n";
	log_file << "\n\nAnticlustering Heuristic\n";

	int l = 0;
	//Start swap heuristic
	//Find points to be swapped
	std::vector<int> changes;
	changes.reserve(n);
	for (int i = 0; i < n; i++)
		for (int c = 0; c < k; ++c)
			if (results.heu_sol(i,c) == 1 and ub_sol(i,c) == 0)
				changes.push_back(i);
	std::cout << "\n\nNum of swaps " << changes.size();
	log_file << "\n\nNum of swaps " << changes.size();

	log_file << "\n\niter | k | W | GAP perc";
	log_file << "\n" << l << " | " << 0 << " | " << std::setprecision(2) << best_W << " | " << std::setprecision(5) <<  best_GAP;

	std::printf("\n\nAnticlustering Heuristic\niter | k | W | GAP perc");
	std::printf("\n%d | %d | %.2f | %.6f", 0, 0, best_W, best_GAP);


	std::ofstream f;
	f.open(result_path + "_swaps.txt");
	for (l = 1; l < num_rep; l++) {

		if (best_GAP < min_gap) {
			log_file << "\n\nMin GAP reached\n";
			std::cout << "\n\nMin GAP reached\n";
			break;
		}

		auto time_h = std::chrono::high_resolution_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(time_h - start_time_h).count() > 1800) {
			log_file << "\n\nTime limit reached\n";
			std::cout << "\n\nTime limit reached\n";
			break;
		}

		bool found_better = false;

		for (int c = 0; c < k; c++) {
			arma::uvec worst_p = arma::sort_index(W_hc.col(c), "ascend");
			for (int idx1 = 0; idx1 < p; idx1++) {

				bool found_better_ch = false;

				// Compute centroid to select nearest point
				int h1 = worst_p(idx1);
				/*
				arma::mat centroid = arma::zeros(1,d);
				arma::mat data_cls = arma::zeros(sizes[c][h1], d);
				int nc = 0;
				for (int i : points[c][h1]) {
					data_cls.row(nc) = data.row(i);
					centroid.row(0) += data.row(i);
					nc++;
				}
				centroid.row(0) /= nc;
				arma::vec dist = arma::sum(arma::square(data_cls.each_row() - centroid),1);
				*/

				// take point with min distance from centroid
				//int idx_p1 = dist.index_min();
				int idx_p1 = 0;
				int p1 = points[c][h1][idx_p1];
				for (int idx2 = p-1; idx2 >= 0 && !found_better_ch; idx2--) {
					if (idx1 != idx2) {
					int h2 = worst_p(idx2);
					for (int idx_p2 = points[c][h2].size()-1; idx_p2 >= 0 && !found_better_ch; idx_p2--) {
						int p2 = points[c][h2][idx_p2];

						// evaluate swap
						arma::mat new_sol = arma::zeros(n,k);
						arma::mat new_antic_sol = antic_sol;
						new_antic_sol(p1,h1)=0;
						new_antic_sol(p1,h2)=1;
						new_antic_sol(p2,h1)=1;
						new_antic_sol(p2,h2)=0;
						arma::mat new_W_hc = evaluate_swap(data, new_antic_sol, centroids_heu, new_sol);
						double W = arma::accu(new_W_hc);
						if (W > best_W and W < results.heu_mss) {
							if (l == 27 or l==1)
								f << "cluster " + std::to_string(c) + " swap " + std::to_string(p1) + " change " + std::to_string(p2) + "\n";
							found_better = true;
							found_better_ch = true;
							best_W = W;
							W_hc = new_W_hc;
							ub_sol = new_sol;
							antic_sol = new_antic_sol;
							update(antic_sol, ub_sol, points, sizes);
							best_GAP = (results.heu_mss - best_W) / results.heu_mss * 100;

							log_file << "\n" << l << " | " << c+1 << " | " << best_W << " | " << best_GAP;
							std::printf("\n%d | %d | %.2f | %.2f", l, c+1, best_W, best_GAP);
						}
					}
				}
				}
			}
		}

		if (!found_better) {
			log_file << "\n\nNo better sol at " << l << "\n\n";
			std::cout << "\n\nNo better sol at " << l << "\n\n";
			break;
		}

    }

    f.close();

	results.anti_obj = best_W;

	std::cout << "\n\nAnticlusters\n";
	std::vector<std::vector<int>> sol(p);
	for (int h = 0; h < p; h++) {
		sol[h].reserve(n);
		std::cout << h << ":  ";
		for (int c = 0; c < k; c++)
			for (int i = 0; i < sizes[c][h]; i++) {
				sol[h].push_back(points[c][h][i]);
				std::cout << " " << points[c][h][i];
			}
		std::cout << "\n";
	}
	log_file << "\n---------------------------------------------------------------------\n";

	// save heuristic time
	auto end_time_h = std::chrono::high_resolution_clock::now();
	results.h_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_h - start_time_h).count();

	/*// print post heu ub
	arma::mat sdp_sol2 = save_ub(data, ub_sol);
	for (int i = 0; i < n; i++)
		for (int h = 0; h < p; h++)
			if (antic_sol(i, h) == 1)
				sdp_sol2(i,0) = h;
	save_to_file(sdp_sol2, "post_heu_UB");
	*/

	// create true lower bound
	auto start_time_lb = std::chrono::high_resolution_clock::now();

    auto *shared_data_part = new SharedDataPartition();

    shared_data_part->threadStates.reserve(n_threads_part);
    for (int i = 0; i < n_threads_part; i++)
        shared_data_part->threadStates.push_back(false);

    for (int h = 0; h < p; ++h) {
        auto *job = new PartitionJob();
        job->part_id = h;

        arma::mat data_part(sol[h].size(), d);
		for (int i = 0; i < sol[h].size(); i++)
			data_part.row(i) = data.row(sol[h][i]);

    	Kmeans kmeans(data_part, k, kmeans_verbose);
    	kmeans.start(1000, 100, 1);
    	job->max_ub = kmeans.objectiveFunction();
        job->part_data = data_part;
        shared_data_part->queue.push_back(job);
        shared_data_part->print = true;

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
    log_file << "\nLB+: " << results.ub_mss;
    log_file << "\nANTI OBJ: " << results.anti_obj;

    log_file << "\n\nInit sol: " << results.heu_mss;
    log_file << "\nNew UB: " << new_ub << "\n\n\n";

    //double p_obj = 0;
	//for (int h = 0; h < p; h++) {
	//	std::cout << "Partition " << h << ": " << arma::accu(W_hc.row(h)) << std::endl;
	//	p_obj += arma::accu(W_hc.row(h));
	//}
	//std::cout << "All p : " << p_obj;

	results.it = l;
	results.heu_mss = new_ub;
	results.heu_sol = sol_ub;
	save_to_file(sdp_sol, "UB");

}


void exact(arma::mat &data, HResult &results) {

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
	std::vector<arma::mat> cls_data(k);
    for (int c = 0; c < k; c++) {
    	int nc = 0;
		cls_data[c] = arma::zeros(n,d);
		for (int i = 0; i < n; ++i) {
    		if (results.heu_sol(i,c) == 1) {
				for (int l = 0; l < d; l++) {
					all_data[i][l] = data(i,l);
					centroid[c][l] += data(i,l);
				}
    			cls[c].push_back(i);
				cls_data[c].row(nc) = data.row(i);
    			nc++;
    		}
    	}
    	cls_data[c] = cls_data[c].rows(0, nc-1);
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
        job->data_cls = cls_data[c];
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
    	mount_model *model = new mount_gurobi_model(env, k*k*p*p/2, all_dist, sol_cls);

    	model->add_point_constraints();
    	model->add_cls_constraints();
    	model->add_edge_constraints();

    	model->optimize();

    	sol = model->get_x_solution(sol_cls);
    	delete env;

    } catch (GRBException &e) {
    	std::cout << "Error code = " << e.getErrorCode() << std::endl;
    	std::cout << e.getMessage() << std::endl;
    }

	// save mount time
	auto end_time_m = std::chrono::high_resolution_clock::now();
	results.m_time += std::chrono::duration_cast<std::chrono::seconds>(end_time_m - start_time_m).count();

	// compute obj
	double sol_obj = 0;
    for (int h = 0; h < p; ++h)
    	for (int i : sol[h])
    		for (int j : sol[h])
    			if (i!=j)
    				sol_obj += (all_dist[i][j] /2) / sol[h].size();
    std:: cout << "\n\n\n\n Sol obj" << sol_obj;


	// create true lower bound
	auto start_time_lb = std::chrono::high_resolution_clock::now();

    auto *shared_data_part = new SharedDataPartition();

    shared_data_part->threadStates.reserve(n_threads_part);
    for (int i = 0; i < n_threads_part; i++)
        shared_data_part->threadStates.push_back(false);

    for (int h = 0; h < p; ++h) {
        auto *job = new PartitionJob();
        job->part_id = h;
        job->max_ub = std::numeric_limits<double>::infinity();

        arma::mat data_part(sol[h].size(), d);
		for (int i = 0; i < sol[h].size(); i++)
			data_part.row(i) = data.row(sol[h][i]);

        job->part_data = data_part;
        shared_data_part->queue.push_back(job);
        shared_data_part->print = true;

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
    log_file << "\nLB+: " << results.ub_mss;
    log_file << "\nANTI OBJ: " << results.anti_obj;

    log_file << "\n\nInit sol: " << results.heu_mss;
    log_file << "\nNew UB: " << new_ub << "\n\n\n";

	results.it = 0;
	results.heu_mss = new_ub;
	results.heu_sol = sol_ub;
	save_to_file(sdp_sol, "UB");

}


arma::mat create_first_sol(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes) {

	arma::mat obj(p,k);
	UserConstraints constraints;
	for (int h = 0; h < p; h++) {

		int nh = arma::accu(antic_sol.col(h));
		arma::mat data_antic(nh,d);
		arma::vec points(nh);
		nh = 0;
		for (int i = 0; i < n; ++i) {
			if (antic_sol(i,h) == 1) {
				data_antic.row(nh) = data.row(i);
				points(nh) = i;
				nh++;
			}
		}

    	Kmeans kmeans(data_antic, k, kmeans_verbose);
    	kmeans.start(kmeans_start, kmeans_max_it, 1);
		arma::mat obj_h(1, k);
		obj_h.row(0) = kmeans.objectiveFunctionCls().t();

		arma::mat sol(nh,k);
		sol = kmeans.getAssignments();
		arma::mat centroids = kmeans.getCentroids();

		std::vector<int> c_idx(k);
		for (int i = 0; i < k; ++i)
			c_idx[i] = i;
		std::vector<int> mapping(k, -1);
		for (int c1 = 0; c1 < k; ++c1) {
			double min_distance = std::numeric_limits<double>::max();
			int best_match = -1;
			int best_idx = -1;
			for (int idx = 0; idx < c_idx.size(); idx++) {
				// Calculate the Euclidean distance between centroids
				int c2 = c_idx[idx];
				double distance = std::pow(arma::norm(centroids_heu.row(c2).t() - centroids.row(c1).t(), 2), 2);
				if (distance < min_distance) {
					min_distance = distance;
					best_match = c2;
					best_idx = idx;
				}
			}
			mapping[c1] = best_match;
			c_idx.erase(c_idx.begin() + best_idx);
		}

		for (int c = 0; c < k; ++c) {
			for (int i = 0; i < nh; i++)
				if (sol(i,c)==1)
					ub_sol(points(i), mapping[c]) = 1;
			obj(h, c) = obj_h(0, mapping[c]);
		}
    }

	//Save new sol as points and sizes
	for (int h = 0; h < p; ++h) {
		for (int c = 0; c < k; c++) {
			points[c][h].reserve(n);
			int nc = 0;
			for (int i = 0; i < n; i++)
				if (antic_sol(i,h)==1 and ub_sol(i,c)==1) {
					points[c][h].push_back(i);
					nc++;
				}
			sizes[c][h] = nc;
		}
	}

    return obj;

}


arma::mat evaluate_swap(arma::mat &data, arma::mat &antic_sol, arma::mat &centroids_heu, arma::mat &new_sol) {

	arma::mat obj(p,k);
	for (int h = 0; h < p; h++) {

		int nh = arma::accu(antic_sol.col(h));
		arma::mat data_antic(nh,d);
		arma::vec points(nh);
		nh = 0;
		for (int i = 0; i < n; ++i) {
			if (antic_sol(i,h) == 1) {
				data_antic.row(nh) = data.row(i);
				points(nh) = i;
				nh++;
			}
		}

		Kmeans kmeans(data_antic, k, kmeans_verbose);
		kmeans.start(kmeans_max_it, 1, centroids_heu);
		obj.row(h) = kmeans.objectiveFunctionCls().t();

		arma::mat sol(nh,k);
		sol = kmeans.getAssignments();
		for (int i = 0; i < nh; ++i) {
			for (int c = 0; c < k; ++c) {
				if (sol(i,c)==1)
					new_sol(points(i), c) = 1;
			}
		}

	}

	return obj;

}

void update(arma::mat &antic_sol, arma::mat &ub_sol, std::vector<std::vector<std::vector<int>>> &points, std::vector<std::vector<int>> &sizes) {

	//Save new sol as points and sizes
	for (int c = 0; c < k; c++) {
		for (int h = 0; h < p; ++h) {
			points[c][h].clear();
			points[c][h].reserve(n);
			int nc = 0;
			for (int i = 0; i < n; i++)
				if (antic_sol(i,h)==1 and ub_sol(i,c)==1) {
					points[c][h].push_back(i);
					nc++;
				}
			sizes[c][h] = nc;
		}
	}

}