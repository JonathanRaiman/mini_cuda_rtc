#ifndef RTC_DEVICE_H
#define RTC_DEVICE_H
struct Device {
    static int gpu_device;
    static void set_gpu_device(int device);
};
#endif
