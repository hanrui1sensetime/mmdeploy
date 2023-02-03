package mmdeploy;

public class Context {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long context_;

    public Context() {
        context_ = create();
    }

    public void add(int contextType, String name, long handle) {
        add(context_, contextType, name, handle);
    }

    public void add(int contextType, long handle) {
        add(context_, contextType, handle);
    }

    public void release() {
        destroy(context_);
    }

    private native long create();

    private native void add(long context, int contextType, String name, long handle);

    private native void add(long context, int contextType, long handle);

    private native void destroy(long context);
}
