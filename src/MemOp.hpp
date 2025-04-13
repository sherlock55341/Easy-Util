#pragma once

#include "Check.hpp"
#include <cstring>

namespace Easy {
void easyCudaMallocBase(void **dev_ptr, size_t size, const char *file,
                        int line);
void easyCudaFreeBase(void *dev_ptr, const char *file, int line);
void easyCudaMemcpyD2DBase(void *dst, void *src, size_t size, const char *file,
                           int line);
void easyCudaMemcpyH2DBase(void *dst, void *src, size_t size, const char *file,
                           int line);
void easyCudaMemcpyD2HBase(void *dst, void *src, size_t size, const char *file,
                           int line);
#define easyCudaMalloc(dev_ptr, size)                                          \
    easyCudaMallocBase(dev_ptr, size, __FILE__, __LINE__)
#define easyCudaFree(dev_ptr) easyCudaFreeBase(dev_ptr, __FILE__, __LINE__)
#define easyCudaMemcpyD2D(dst, src, size)                                      \
    easyCudaMemcpyD2DBase(dst, src, size, __FILE__, __LINE__)
#define easyCudaMemcpyH2D(dst, src, size)                                      \
    easyCudaMemcpyH2DBase(dst, src, size, __FILE__, __LINE__)
#define easyCudaMemcpyD2H(dst, src, size)                                      \
    easyCudaMemcpyD2HBase(dst, src, size, __FILE__, __LINE__)

enum class EasyDev { CPU, CUDA };

inline const char *to_str(EasyDev dev) {
    if (dev == EasyDev::CPU)
        return "CPU";
    if (dev == EasyDev::CUDA)
        return "CUDA";
    return "UNKNOWN";
}

template <typename T> class EasyPtr {
  protected:
    T *_ptr;
    size_t _size;
    EasyDev _dev;

  public:
    EasyPtr() : _ptr(nullptr), _size(0), _dev(EasyDev::CPU) {}
    ~EasyPtr() {
        if (_ptr) {
            if (_dev == EasyDev::CPU)
                ::free(_ptr);
            if (_dev == EasyDev::CUDA)
                easyCudaFreeBase(_ptr, __builtin_FILE(), __builtin_LINE());
            _ptr = nullptr;
            _dev = EasyDev::CPU;
            _size = 0;
        }
    }
    EasyPtr(const EasyPtr &other) = delete;

    T *data() const { return _ptr; }
    T *data(EasyDev dev_expected) const {
        if (dev_expected != _dev) {
            printf(
                "[Data Error] : You are trying to access a slice of memory at "
                "%s:%d, but the device is %s, while you are expecting %s\n",
                __builtin_FILE(), __builtin_LINE(), to_str(_dev),
                to_str(dev_expected));
            exit(-1);
        }
        return _ptr;
    }
    size_t size() const { return _size; }
    EasyDev dev() const { return _dev; }

    void alloc(EasyDev device, size_t N) {
        if (_ptr) {
            printf("[Alloc Error] : You have already held a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        _dev = device;
        _size = N;
        if (device == EasyDev::CPU)
            _ptr = (T *)malloc(sizeof(T) * N);
        if (device == EasyDev::CUDA)
            easyCudaMallocBase((void **)&_ptr, sizeof(T) * N, __builtin_FILE(),
                               __builtin_LINE());
    }
    void free() {
        if (_ptr == nullptr) {
            printf(
                "[Free Error] : You do not hold a slice of memory at %s:%d\n",
                __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_dev == EasyDev::CPU)
            ::free(_ptr);
        if (_dev == EasyDev::CUDA)
            easyCudaFreeBase(_ptr, __builtin_FILE(), __builtin_LINE());
        _ptr = nullptr;
        _size = 0;
        _dev = EasyDev::CPU;
    }

    void to(const EasyPtr<T> &other) const {
        T *src = _ptr;
        T *dst = other._ptr;
        if (src == nullptr) {
            printf("[Memcpy Src Error] : You do not hold a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (dst == nullptr) {
            printf("[Memcpy Dst Error] : You do not hold a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_size != other._size) {
            printf("[Memcpy Size Error] : Src and dst have different sizes at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_dev == EasyDev::CPU && other._dev == EasyDev::CPU)
            memcpy(dst, src, sizeof(T) * _size);
        if (_dev == EasyDev::CUDA && other._dev == EasyDev::CUDA)
            easyCudaMemcpyD2DBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
        if (_dev == EasyDev::CUDA && other._dev == EasyDev::CPU)
            easyCudaMemcpyD2HBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
        if (_dev == EasyDev::CPU && other._dev == EasyDev::CUDA)
            easyCudaMemcpyH2DBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
    }

    void from(const EasyPtr<T> &other) const {
        T *src = other._ptr;
        T *dst = _ptr;
        if (src == nullptr) {
            printf("[Memcpy Src Error] : You do not hold a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (dst == nullptr) {
            printf("[Memcpy Dst Error] : You do not hold a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_size != other._size) {
            printf("[Memcpy Size Error] : Src and dst have different sizes at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_dev == EasyDev::CPU && other._dev == EasyDev::CPU)
            memcpy(dst, src, sizeof(T) * _size);
        if (_dev == EasyDev::CUDA && other._dev == EasyDev::CUDA)
            easyCudaMemcpyD2DBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
        if (_dev == EasyDev::CUDA && other._dev == EasyDev::CPU)
            easyCudaMemcpyH2DBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
        if (_dev == EasyDev::CPU && other._dev == EasyDev::CUDA)
            easyCudaMemcpyD2HBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
    }

    void from(T *src, size_t N, EasyDev src_dev) const {
        T *dst = _ptr;
        if (src == nullptr) {
            printf("[Memcpy Src Error] : You do not hold a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (dst == nullptr) {
            printf("[Memcpy Dst Error] : You do not hold a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_size != N) {
            printf("[Memcpy Size Error] : Src and dst have different sizes at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_dev == EasyDev::CPU && src_dev == EasyDev::CPU)
            memcpy(dst, src, sizeof(T) * _size);
        if (_dev == EasyDev::CUDA && src_dev == EasyDev::CUDA)
            easyCudaMemcpyD2DBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
        if (_dev == EasyDev::CUDA && src_dev == EasyDev::CPU)
            easyCudaMemcpyH2DBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
        if (_dev == EasyDev::CPU && src_dev == EasyDev::CUDA)
            easyCudaMemcpyD2HBase(dst, src, sizeof(T) * _size, __builtin_FILE(),
                                  __builtin_LINE());
    }

    T read(size_t pos) const {
        if (_ptr == nullptr) {
            printf("[Read Error] : You do not hold a slice of memory at "
                   "%s:%d\n",
                   __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (pos >= _size) {
            printf("[Read Error] : Pos %zu is out of range at "
                   "%s:%d\n",
                   pos, __builtin_FILE(), __builtin_LINE());
            exit(-1);
        }
        if (_dev == EasyDev::CPU)
            return _ptr[pos];
        else {
            T tmp;
            easyCudaMemcpyD2HBase(&tmp, _ptr + pos, sizeof(T), __builtin_FILE(),
                                  __builtin_LINE());
            return tmp;
        }
    }

    T &operator[](size_t pos) { return _ptr[pos]; }
};
} // namespace Easy