#ifndef CLUSTERING_THREADPOOLANTI_H
#define CLUSTERING_THREADPOOLANTI_H

#include "sdp_branch_and_bound.h"
#include "util.h"
#include "matlab_util.h"

typedef struct AntiJob {

    int cls_id;
    std::vector<int> cls_points;

} AntiJob;

typedef struct InputDataAnti {

    double max_d;
    arma::mat data;
    std::vector<std::vector<double>> all_dist;

} InputDataAnti;



typedef struct SharedDataAnti {

    // Between workers and main
    std::condition_variable mainConditionVariable;
    std::vector<bool> threadStates;

    // Queue of requests waiting to be processed
    std::deque<AntiJob *> queue;
    // This condition variable is used for the threads to wait until there is work to do
    std::condition_variable queueConditionVariable;
    // Mutex to protect queue
    std::mutex queueMutex;

    std::vector<double> dist_cls; // used to store the objective function of each sub-problem
    std::unordered_map<int, std::unordered_map<int, arma::mat>> sol_cls;

} SharedDataAnti;

class ThreadPoolAnti {

private:

    InputDataAnti  *input_data;
    SharedDataAnti  *shared_data;

    // We store the threads in a vector, so we can later stop them gracefully
    std::vector<std::thread> threads;

    // This will be set to true when the thread pool is shutting down. This tells
    // the threads to stop looping and finish
    bool done;

    void doWork(int id);


public:

    ThreadPoolAnti(InputDataAnti *input_data, SharedDataAnti *shared_data, int n_thread);
    void quitPool();
    void addJob(AntiJob *antiJob);

};


#endif //CLUSTERING_THREADPOOLANTI_H
