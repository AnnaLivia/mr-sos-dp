#include <thread>
#include <condition_variable>
#include <vector>
#include "ThreadPoolPartition.h"
#include "config_params.h"
#include "ac_heuristic.h"


ThreadPoolPartition::ThreadPoolPartition(SharedDataPartition *shared_data, int n_thread) {

    // This returns the number of threads supported by the system.
    // auto numberOfThreads = std::thread::hardware_concurrency();
    int numberOfThreads = n_thread;

    this->shared_data = shared_data;

    done = false;

    for (int i = 0; i < numberOfThreads; ++i) {
        // The threads will execute the private member `doWork`. Note that we need
        // to pass a reference to the function (namespaced with the class name) as
        // the first argument, and the current object as second argument
        threads.emplace_back(&ThreadPoolPartition::doWork, this, i);
    }

}

// The destructor joins all the threads so the program can exit gracefully.
void ThreadPoolPartition::quitPool() {

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
void ThreadPoolPartition::addJob(PartitionJob *job) {
    // Grab the mutex
    std::lock_guard<std::mutex> l(shared_data->queueMutex);

    // Push the request to the queue
    shared_data->queue.push_back(job);

    // Notify one thread that there are requests to process
    shared_data->queueConditionVariable.notify_one();
}

// Function used by the threads to grab work from the queue
void ThreadPoolPartition::doWork(int id) {

    while (true) {

        PartitionJob *job;

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

        int np = (int) job->part_data.n_rows;
        std::cout << std::endl << "*********************************************************************" << std::endl;
        std::cout << "Partition " << (job->part_id + 1) << " processed by Thread "<< id << "\nPoints " << np;
        std::cout << std::endl << "*********************************************************************" << std::endl;

        // Lock the mutex to ensure thread-safe access to the shared global file stream
        std::lock_guard<std::mutex> guard(file_mutex);

        log_file << "\nPARTITION " << (job->part_id + 1);
        arma::mat sol(np, k);
        UserConstraints constraints;

        /*
        // standardize data
        double lb_mssc;
        if (stddata) {
            arma::mat standardized = job->part_data;
            arma::mat dist = compute_distances(standardized);
            arma::rowvec means = arma::mean(standardized, 0);
            arma::rowvec stddevs = arma::stddev(standardized, 0, 0);
            standardized.each_row() -= means;
            standardized.each_row() /= stddevs;
            sdp_branch_and_bound(k, standardized, constraints, sol);
            lb_mssc = compute_mss(job->part_data, sol);
        }
        else
        */
        double ub_mssc = 0;
        double lb_mssc = sdp_branch_and_bound(k, job->part_data, ub_mssc, constraints, sol, job->max_ub, shared_data->print);
        arma::mat sol_f = std::move(arma::join_horiz(job->part_data, sol));
        //save_to_file(sol_f, "LB_" + std::to_string(job->part_id + 1));

        arma::mat cls(np, d+1);
        for (int i = 0; i < np; i++) {
            for (int j = 0; j < d; j++)
                cls(i,j) = job->part_data(i,j);
            for (int c = 0; c < k; c++)
                if (sol(i,c) == 1)
                    cls(i,d) = c;
        }

        {
            std::lock_guard<std::mutex> l(shared_data->queueMutex);
            shared_data->lb_part.push_back(lb_mssc);
            shared_data->ub_part.push_back(ub_mssc);
            shared_data->sol_part[job->part_id] = cls;
            shared_data->threadStates[id] = false;
        }

        shared_data->mainConditionVariable.notify_one();

        delete (job);

    }
}
