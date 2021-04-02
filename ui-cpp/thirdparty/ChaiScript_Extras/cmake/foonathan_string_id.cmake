set(STRING_ID_VERSION 6e2e5c48ee4a3ac0c54ba505f0a573561f2979ec)
find_package(foonathan_string_id 2.0.3 QUIET)

if (NOT foonathan_string_id_FOUND)
  include(FetchContent)

  FetchContent_Declare(
    foonathan_string_id
    GIT_REPOSITORY https://github.com/foonathan/string_id.git
    GIT_TAG ${STRING_ID_VERSION}
  )

  FetchContent_GetProperties(foonathan_string_id)
  if (NOT foonathan_string_id_POPULATED)
    set(FETCHCONTENT_QUIET NO)
    FetchContent_Populate(foonathan_string_id)

    add_subdirectory(${foonathan_string_id_SOURCE_DIR} ${foonathan_string_id_BINARY_DIR})
  endif()
endif()
