#ifndef CHAISCRIPT_EXTRAS_STRING_ID_HPP_
#define CHAISCRIPT_EXTRAS_STRING_ID_HPP_

#include <cmath>
#include <memory>

#include <chaiscript/chaiscript.hpp>
#include <string_id/string_id.hpp>
#include <string_id/database.hpp>
#include <string_id/basic_database.hpp>

namespace chaiscript {
  namespace extras {
    namespace string_id {

      ModulePtr bootstrap(ModulePtr m = std::make_shared<Module>())
      {

        using namespace foonathan;

        // default_database
        m->add(user_type<foonathan::string_id::default_database>(), "default_database");
        m->add(constructor<foonathan::string_id::default_database ()>(), "default_database");

        // basic_database
#ifdef FOONATHAN_STRING_ID_DATABASE
        m->add(user_type<foonathan::string_id::basic_database>(), "basic_database");
        m->add(base_class<foonathan::string_id::basic_database, foonathan::string_id::default_database>());
#endif

        // string_id
        m->add(user_type<foonathan::string_id::string_id>(), "string_id");
        m->add(constructor<foonathan::string_id::string_id (const foonathan::string_id::string_id &)>(), "string_id");
        m->add(constructor<foonathan::string_id::string_id (const foonathan::string_id::string_info &, foonathan::string_id::basic_database &)>(), "string_id");
        m->add(constructor<foonathan::string_id::string_id (const foonathan::string_id::string_id &, foonathan::string_id::string_info)>(), "string_id");
        m->add(fun(&foonathan::string_id::string_id::hash_code), "hash_code");
        m->add(fun(&foonathan::string_id::string_id::database), "database");
        m->add(fun<bool (const foonathan::string_id::string_id &, const foonathan::string_id::string_id &)>([](const foonathan::string_id::string_id &t_lhs, const foonathan::string_id::string_id &t_rhs){ return t_lhs == t_rhs; }), "==");
        m->add(fun<bool (foonathan::string_id::hash_type, const foonathan::string_id::string_id &)>([](foonathan::string_id::hash_type t_lhs, const foonathan::string_id::string_id &t_rhs){ return t_lhs == t_rhs; }), "==");
        m->add(fun<bool (const foonathan::string_id::string_id &, foonathan::string_id::hash_type)>([](const foonathan::string_id::string_id &t_lhs, foonathan::string_id::hash_type t_rhs){ return t_lhs == t_rhs; }), "==");
        m->add(fun<bool (const foonathan::string_id::string_id &, const foonathan::string_id::string_id &)>([](const foonathan::string_id::string_id &t_lhs, const foonathan::string_id::string_id &t_rhs){ return t_lhs != t_rhs; }), "!=");
        m->add(fun<bool (foonathan::string_id::hash_type, const foonathan::string_id::string_id &)>([](foonathan::string_id::hash_type t_lhs, const foonathan::string_id::string_id &t_rhs){ return t_lhs != t_rhs; }), "!=");
        m->add(fun<bool (const foonathan::string_id::string_id &, foonathan::string_id::hash_type)>([](const foonathan::string_id::string_id &t_lhs, foonathan::string_id::hash_type t_rhs){ return t_lhs != t_rhs; }), "!=");

        // string_info
        m->add(user_type<foonathan::string_id::string_info>(), "string_info");
        m->add(fun(&foonathan::string_id::string_info::string), "string");
        m->add(fun(&foonathan::string_id::string_info::length), "length");



        // type conversions
        m->add(type_conversion<const std::string &, foonathan::string_id::string_info>(
              [](const std::string &t_str) { 
                return foonathan::string_id::string_info(t_str.c_str(), t_str.size()); 
              }
            ));

        // free functions
        m->add(fun(&foonathan::string_id::detail::sid_hash), "sid_hash");

        return m;
      }
    }
  }
}

#endif /* CHAISCRIPT_EXTRAS_STRING_ID_HPP_ */