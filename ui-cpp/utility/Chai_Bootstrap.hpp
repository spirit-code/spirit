#pragma once
#ifndef CHAI_BOOTSTRAP_HPP
#define CHAI_BOOTSTRAP_HPP

#include <Spirit/System.h>
#include <Spirit/Geometry.h>
#include <chaiscript/chaiscript.hpp>
#include <array>
#include <vector>

namespace Utility{

    template<typename T>
    class Vector_View
    {
        public:
        Vector_View() : m_buf(nullptr), N(0) {};
        Vector_View(T* buf, int N) : m_buf(buf), N(N) {};

        const T & operator[](const int i) const
        {
            if(i>=0 && i<N)
            {
                return m_buf[i];
            } else {
                throw chaiscript::exception::eval_error("Access outside of range!");
            }
        }

        int size() const
        {
            return N;
        }

        private:
        T * m_buf;
        int N;
    };

    class Spirit_Chai
    {
        using Vector3 = std::array<double, 3>;

        public:
        Spirit_Chai();

        void update(State * p_state);

        chaiscript::ChaiScript & get();
        void reset_state();
        chaiscript::ChaiScript::State state();
        void set_state(const chaiscript::ChaiScript::State & state);

        private:
        chaiscript::ChaiScript chai;
        chaiscript::ChaiScript::State chai_state;
        std::array<int, 3> n_cells = {0,0,0};
        int n_cell_atoms = 0;
        Vector3 a_vec    = {0,0,0};
        Vector3 b_vec    = {0,0,0};
        Vector3 c_vec    = {0,0,0};
        Vector3 center   = {0,0,0};

        Vector_View<Vector3> spins;
        Vector_View<Vector3> positions;
    };
};
#endif