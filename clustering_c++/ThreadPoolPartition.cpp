#include <thread>
#include <condition_variable>
#include <vector>
#include "ThreadPoolPartition.h"
#include "config_params.h"


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

        UserConstraints constraints;
        int np = (int) job->part.n_rows;
        std::cout << std::endl << "*********************************************************************" << std::endl;
        std::cout << "Partition " << (job->part_id + 1) << " processed by Thread "<< id << "\nPoints " << np;
        std::cout << std::endl << "*********************************************************************" << std::endl;
        log_file << "Partition " << (job->part_id + 1) << "\n";
        arma::mat data = job->part.submat(0, 1, np-1, d);
        arma::mat sol(np, k);
        double lb_mssc = sdp_branch_and_bound(k, data, constraints, sol);

        arma::mat cls(np, 1);
        for (int i = 0; i < np; i++)
            for (int c = 0; c < k; c++)
                if (sol(i,c) == 1)
                    cls(i)= c + 1;

        arma::mat sol_part = arma::join_horiz(job->part, cls);

        delete (job);

        {
            std::lock_guard<std::mutex> l(shared_data->queueMutex);
            shared_data->lb_part.push_back(lb_mssc);
            shared_data->sol_part.push_back(sol_part);
            shared_data->threadStates[id] = false;
        }

        shared_data->mainConditionVariable.notify_one();

    }
}
