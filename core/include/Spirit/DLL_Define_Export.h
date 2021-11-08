/*
 * This header, included at the start of API headers,
 * defines the necessary export keyword
 */

// clang-format off
#ifdef _WIN32

    #ifdef __cplusplus
        #define PREFIX extern "C" __declspec(dllexport)
        #define SUFFIX noexcept
    #else
        #define PREFIX __declspec(dllexport)
        #define SUFFIX
    #endif

#else

    #ifdef __cplusplus
        #define PREFIX extern "C"
        #define SUFFIX noexcept
    #else
        #define PREFIX
        #define SUFFIX
    #endif

#endif
// clang-format on
