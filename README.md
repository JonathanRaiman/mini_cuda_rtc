Mini CUDA RTC
-------------

[Runtime Compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) is used in this mini CUDA Array library to create at runtime new kernels based on user input. You can for instance ask a kernel to perform addition, subtraction, or multiplication when the relevant code was never present before. The library will call up NVCC and load up a module containing the new code. On later calls the code generated will be either reloaded or simply re-used if possible.

Using this technique it should be feasible to reduce compilation times for large CUDA projects by making the runtime decide what are the right pieces of code to generate.

Note: this is a proof of concept and not a fully-fledged arsenal of runtime operations.

### Usage

You can choose what kernel gets run by changing the argument to the function:

1. Without arguments this defaults to addition:

    ```bash
    ./simple_rtc_nvcc
    ```

1. Call multiplication:

    ```bash
    ./simple_rtc_nvcc "*"
    ```

1. Call division:

    ```bash
    ./simple_rtc_nvcc "/"
    ```

1. Call addition:

    ```bash
    ./simple_rtc_nvcc +
    ```

### Installation

Ensure you have a cuda-capable device available along with `nvcc` installed.

```bash
nvcc -std=c++11 simple_rtc_nvcc.cu -o simple_rtc_nvcc
```

