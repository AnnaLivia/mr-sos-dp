#include <thread>
#include <condition_variable>
#include <vector>
#include "ThreadPoolAnti.h"
#include "config_params.h"
#include "comb_model.h"
#include "norm1_model.h"
#include "ac_heuristic.h"


ThreadPoolAnti::ThreadPoolAnti(SharedDataAnti *shared_data, int n_thread) {

    // This returns the number of threads supported by the system.
    // auto numberOfThreads = std::thread::hardware_concurrency();
    int numberOfThreads = n_thread;

    this->shared_data = shared_data;

    done = false;

    for (int i = 0; i < numberOfThreads; ++i) {
        // The threads will execute the private member `doWork`. Note that we need
        // to pass a reference to the function (namespaced with the class name) as
        // the first argument, and the current object as second argument
        threads.emplace_back(&ThreadPoolAnti::doWork, this, i);
    }

}

// The destructor joins all the threads so the program can exit gracefully.
void ThreadPoolAnti::quitPool() {

    {
        std::lock_guard<std::mutex> l(shared_data->queueMutex);
        // So threads know it's time to shut down
        done = true;
    }

    // Wake up all the threads, so they can finish and be joined
    shared_data->queueConditionVariable.notify_all();

    for (auto& thread : threads) {
        thread.join();
    }
}

// This function will be called by the server every time there is a request
// that needs to be processed by the thread pool
void ThreadPoolAnti::addJob(AntiJob *job) {
    // Grab the mutex
    std::lock_guard<std::mutex> l(shared_data->queueMutex);

    // Push the request to the queue
    shared_data->queue.push_back(job);

    // Notify one thread that there are requests to process
    shared_data->queueConditionVariable.notify_one();
}

// Function used by the threads to grab work from the queue
void ThreadPoolAnti::doWork(int id) {

    while (true) {

        AntiJob *job;

        {
            std::unique_lock<std::mutex> l(shared_data->queueMutex);
            while (shared_data->queue.empty() && !done) {
                // Only wake up if there are elements in the queue or the program is shutting down
                shared_data->queueConditionVariable.wait(l);
            }


            // If we are shutting down exit without trying to process more work
            if (done) break;


            shared_data->threadStates[id] = true;


            job = shared_data->queue.front();
            shared_data->queue.pop_front();
        }

        // run anti
        std::cout << std::endl << "*********************************************************************" << std::endl;
        std::cout << "Cluster " << job->cls_id + 1 << " processed by Thread "<< id << "\nPoints " << job->cls_points.size();
        std::cout << std::endl << "*********************************************************************" << std::endl;

        //std::pair<double, std::vector<std::vector<int>>> sol = compute_anti_cls(job->cls_points, shared_data->all_dist);
		std::pair<double, std::vector<std::vector<int>>> sol;

    	try {
    		GRBEnv *env = new GRBEnv();
    		int nc = (int) job->cls_points.size();
    		//comb_model *model = new comb_gurobi_model(env, nc, p, d, job->cls_points);
    		norm_model *model = new norm_gurobi_model(env, nc, p, d, job->cls_points);

	    	model->add_point_constraints();
    		model->add_part_constraints();
		    model->add_dev_constraints(shared_data->all_data, job->center);

	    	model->optimize();

	        // Lock the mutex to ensure thread-safe access to the shared global file stream
    	    std::lock_guard<std::mutex> guard(file_mutex);

    		log_file << "Cluster " << job->cls_id + 1 << ": " << model->get_gap() * 100 << "%" << "\n";
    		std::vector<std::vector<int>> cls_ass = model->get_x_solution();
    		sol = std::make_pair(model->get_value(), cls_ass);

	//    	delete model;
    		delete env;

	    } catch (GRBException &e) {
    		std::cout << "Error code = " << e.getErrorCode() << std::endl;
    		std::cout << e.getMessage() << std::endl;
    	}


        double best_dist = sol.first;
        std::vector<std::vector<int>> best_part_points = sol.second;

        {
            std::lock_guard<std::mutex> l(shared_data->queueMutex);
            shared_data->dist_cls.push_back(best_dist);

            // update final partition
            for (int h = 0; h < p; ++h)
                shared_data->sol_cls[job->cls_id][h] = best_part_points[h];

            shared_data->threadStates[id] = false;
        }

        shared_data->mainConditionVariable.notify_one();

        delete (job);

    }
}
