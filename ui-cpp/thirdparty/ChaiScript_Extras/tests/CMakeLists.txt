list(APPEND LIBS ${READLINE_LIB})

# Add catch tests macro
macro(ADD_CATCH_TESTS executable)
  if (MSVC)
    file(TO_NATIVE_PATH "${QT_LIBRARY_DIR}" QT_LIB_PATH)
    set(NEWPATH "${QT_LIB_PATH};$ENV{PATH}")
  else()
    set(NEWPATH $ENV{PATH})
  endif()

  get_target_property(target_files ${executable} SOURCES)

  message("Files: ${target_files}")

  foreach(source ${target_files})
    if(NOT "${source}" MATCHES "/moc_.*cxx")
      string(REGEX MATCH .*cpp source "${source}")
      if(source)
        file(READ "${source}" contents)
        string(REGEX MATCHALL "TEST_CASE\\([ ]*\"[^\"]+\"" found_tests ${contents})
        foreach(hit ${found_tests})
          message("Found Test: ${hit}")
          string(REGEX REPLACE "TEST_CASE\\([ ]*(\"[^\"]+\").*" "\\1" test_name ${hit})
          add_test(${test_name} "${executable}" ${test_name})
          set_tests_properties(${test_name} PROPERTIES TIMEOUT 660 ENVIRONMENT "PATH=${NEWPATH}")
        endforeach()
      endif()
    endif()
  endforeach()
endmacro()

# Math
add_executable(math_test math.cpp)
target_link_libraries(math_test ${LIBS})
target_include_directories(math_test PUBLIC "${chaiscript_SOURCE_DIR}/include")
ADD_CATCH_TESTS(math_test)

# String ID
#add_executable(string_id_test string_id.cpp)
#target_link_libraries(string_id_test ${LIBS} foonathan_string_id)
#target_include_directories(string_id_test PUBLIC
#	"${chaiscript_SOURCE_DIR}/include"
#	"${foonathan_string_id_SOURCE_DIR}"
#)
#ADD_CATCH_TESTS(string_id_test)

# String Methods
add_executable(string_methods_test string_methods.cpp)
target_link_libraries(string_methods_test ${LIBS})
target_include_directories(string_methods_test PUBLIC "${chaiscript_SOURCE_DIR}/include")
ADD_CATCH_TESTS(string_methods_test)
