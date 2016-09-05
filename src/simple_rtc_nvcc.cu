#include "config.h" // contains information about header location
#include "array.h"
#include "array-impl.h"
#include "array_gather.h"
#include "array_gather-impl.h"
#include "timer.h"
#include "compiler.h"
#include <string>

#define STR(x) __THIS_IS_VERY_ABNOXIOUS(x)
#define __THIS_IS_VERY_ABNOXIOUS(tok) #tok

std::function<void(ArrayGather<float>, Array<float>)> get_func_with_operator(
        Compiler& compiler,
        std::string operator_name) {
    std::string code = (
        "struct CustomSaver {\n"
        "    template<typename T>\n"
        "    static void XINLINE save(T& left, const T& right) {\n"
        "        left " + operator_name + "= right;\n"
        "    }\n"
        "};\n"
        "\n"
        "template<typename SrcT>\n"
        "void rtc_func(ArrayGather<SrcT> source,\n"
        "              const Array<SrcT>& updates,\n"
        "              cudaStream_t stream = NULL) {\n"
        "    scatter_saver<CustomSaver>(\n"
        "        source.array_, source.indices_, updates, stream\n"
        "    );\n"
        "};"
    );
    return compiler.compile<ArrayGather<float>, Array<float>>(code, "rtc_func", true);
}

template class Array<float>;
template class Array<int>;

int main(int argc, char** argv) {
    int dim = 5;
    int cols = 3;

    Array<float> source({dim, cols}, DEVICE_T_GPU);
    source = 0;

    Array<float> updates({dim, cols}, DEVICE_T_GPU);
    updates = 2;

    Array<int> indices({dim}, DEVICE_T_CPU);
    int i = 0;
    for (auto index : {0, 0, 2, 1, 2}) {
        indices[i++] = index;
    }
    indices.print();
    auto indices_gpu = indices.to_device(DEVICE_T_GPU);
    source.print();
    // increment repeatedly at this location:
    source[indices_gpu] += updates;
    source.print();
    // decrement repeatedly at this location:
    source[indices_gpu] -= updates;
    source.print();
    // not well defined in many to one setup
    source[indices_gpu] = updates;
    source.print();

    // RTC
    std::string operator_name = "+";
    // choose symbol at runtime
    if (argc > 1) {
        operator_name = argv[1];
    }

    Compiler compiler(
        {Headerfile(STR(PROJECT_DIR) "/src/array.h", "array.h"),
         Headerfile(STR(PROJECT_DIR) "/src/array_gather.h", "array_gather.h"),
         Headerfile(STR(PROJECT_DIR) "/src/kernels.h", "kernels.h")},
        STR(PROJECT_DIR) "/src",
        "/tmp"
    );

    // run functor defined by user at runtime:
    auto func = get_func_with_operator(compiler, operator_name);
    func(source[indices_gpu], updates);
    source.print();

    // decrement repeatedly at this location:
    auto func2 = get_func_with_operator(compiler, "-");
    func2(source[indices_gpu], updates);
    source.print();

    // run functor defined by user at again:
    func(source[indices_gpu], updates);
    source.print();

    Timer::report();

    return EXIT_SUCCESS;
}
