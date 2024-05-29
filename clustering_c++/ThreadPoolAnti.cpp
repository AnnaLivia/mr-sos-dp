#include <thread>
#include <condition_variable>
#include <vector>
#include "ThreadPoolAnti.h"
#include "config_params.h"
#include "ac_heuristic.h"


ThreadPoolAnti::ThreadPoolAnti(InputDataAnti *input_data, SharedDataAnti *shared_data, int n_thread) {

    // This returns the number of threads supported by the system.
    // auto numberOfThreads = std::thread::hardware_concurrency();
    int numberOfThreads = n_thread;

    this->input_data = input_data;
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
        std::pair<double, std::unordered_map<int, std::vector<int>>> sol = compute_anti_single_cluster(job->cls_points,
                                                                                                        input_data->max_d, input_data->all_dist);

        double best_dist = sol.first;
        std::unordered_map<int, std::vector<int>> best_part_map = sol.second;

        {
            std::lock_guard<std::mutex> l(shared_data->queueMutex);

            shared_data->dist_cls.push_back(best_dist);

            // update sol map
            for (int h = 0; h < p; ++h) {
                shared_data->sol_cls[job->cls_id][h] = arma::mat(job->cls_points.size(), d + 1);
                int np = 0;
                for (int i : best_part_map[h]) {
                    shared_data->sol_cls[job->cls_id][h](np,0) = i + 1;
                    shared_data->sol_cls[job->cls_id][h].row(np).subvec(1, d) = input_data->data.row(i);
                    np++;
                }
                shared_data->sol_cls[job->cls_id][h] = shared_data->sol_cls[job->cls_id][h].submat(0, 0, np - 1, d);
            }


            shared_data->threadStates[id] = false;
        }

        shared_data->mainConditionVariable.notify_one();

        delete (job);

    }
}
