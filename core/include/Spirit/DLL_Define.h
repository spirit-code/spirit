/*
 * This header, included at the start of API headers,
 * defines the necessary export and deprecated macros
 */

// clang-format off
#ifdef _WIN32
    #define WIN_PREFIX __declspec(dllexport)
#else
    #define WIN_PREFIX
#endif

#ifdef __cplusplus
    #define PREFIX extern "C" WIN_PREFIX
    #define SUFFIX noexcept
#else
    #define PREFIX WIN_EXPORT
    #define SUFFIX
#endif

#if defined( __cplusplus )
    // Standard for C++14 and later
    #define DEPRECATED( msg ) extern "C" [[deprecated( msg )]] WIN_PREFIX
#elif defined( __STDC_VERSION__ ) && __STDC_VERSION__ >= 202311L
    // Standard for C23
    #define DEPRECATED( msg ) [[deprecated( msg )]]
#elif defined( __GNUC__ ) || defined( __clang__ )
    #define DEPRECATED( msg ) __attribute__( ( deprecated( msg ) ) )
#elif defined( _MSC_VER )
    #define DEPRECATED( msg ) __declspec( deprecated( msg ) )
#else
    #define DEPRECATED( msg )
#endif

// clang-format on
