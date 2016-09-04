#include "device.h"
#include <stdexcept>
#include <iostream>
#include "make_message.h"

void Device::set_gpu_device(int device) {
    if (gpu_device != device) {
        int count;
        cudaGetDeviceCount(&count);
        if (count == 0) {
            throw std::runtime_error(
                "no gpu devices found"
            );
        } else {
            std::cout << count << " gpu device" << (count == 1 ? " " : "s ") << "found" << std::endl;
        }

        auto status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                make_message(
                    "could not set the gpu device to ",
                    device, ", reason = ", cudaGetErrorString(status)
                )
            );
        }
        gpu_device = device;
    }
}

int Device::gpu_device = -1;
