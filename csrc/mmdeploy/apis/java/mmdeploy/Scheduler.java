package mmdeploy;

public class Scheduler {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long scheduler_;

    public Scheduler(long scheduler) {
        scheduler_ = scheduler;
    }

    public static long threadPool(int numThreads) {
        scheduler_ = createThreadPool(numThreads);
    }

    public static long thread() {
        scheduler_ = createThread();
    }

    public void release() {
        destroy(scheduler_);
    }

    private native long createThreadPool(int numThreads);

    private native long createThread();

    private native void destroy(long scheduler_);
}
