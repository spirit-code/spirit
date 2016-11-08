/*
    This header, included at the start of API headers,
    defines the necessary export keyword
*/
#ifdef _WIN32

    #ifdef __cplusplus
        #define DLLEXPORT extern "C" __declspec(dllexport)
	#else
        #define DLLEXPORT __declspec(dllexport)
    #endif

#else

    #ifdef __cplusplus
        #define DLLEXPORT extern "C"
    #else
        #define DLLEXPORT
    #endif

#endif