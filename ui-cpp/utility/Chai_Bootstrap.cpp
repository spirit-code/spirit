#include <Chai_Bootstrap.hpp>
#include <chaiscript/extras/math.hpp>

namespace Utility
{
    Spirit_Chai::Spirit_Chai()
    {
        auto mathlib = chaiscript::extras::math::bootstrap();
        chai.add(mathlib);

        // Usage of Vector3
        auto get       = [](const Vector3 & vec, int i) { return vec[i]; };
        auto add       = [](const Vector3 & vec1, const Vector3 & vec2) { return Vector3{vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2]}; };
        auto subtract  = [](const Vector3 & vec1, const Vector3 & vec2) { return Vector3{vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]}; };
        auto norm      = [](const Vector3 & vec) { return std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]); };
        auto to_string = [](const Vector3 & vec) { return "[" + std::to_string(vec[0]) + "," + std::to_string(vec[1])+ "," + std::to_string(vec[2]) + "]"; };
        auto dot       = [](const Vector3 & vec1, const Vector3 & vec2) { return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]; };
        auto mult      = [](const Vector3 & vec, float scalar) { return Vector3{scalar * vec[0], scalar * vec[1], scalar * vec[2]}; };
        auto mult2     = [](float scalar, const Vector3 & vec) { return Vector3{scalar * vec[0], scalar * vec[1], scalar * vec[2]}; };
        auto div       = [](const Vector3 & vec, float scalar) { return Vector3{vec[0]/scalar, vec[1]/scalar, vec[2]/scalar}; };
        auto normalize = [div, norm](const Vector3 & vec) { return div(vec, norm(vec)); };
        auto cross     = [](const Vector3 & vec1, const Vector3 & vec2) { return Vector3{ vec1[1]*vec2[2] - vec1[2]*vec2[1], vec1[2]*vec2[0] - vec1[0]*vec2[2], vec1[0]*vec2[1] - vec1[1]*vec2[0] }; };

        chai.add( chaiscript::fun( get ), "[]");
        chai.add( chaiscript::fun( mult ), "*");
        chai.add( chaiscript::fun( mult2 ), "*");
        chai.add( chaiscript::fun( add ), "+");
        chai.add( chaiscript::fun( subtract ), "-");
        chai.add( chaiscript::fun( div ), "/");

        chai.add( chaiscript::fun( norm ), "norm");
        chai.add( chaiscript::fun( normalize ), "normalize");
        chai.add( chaiscript::fun( dot ), "dot");
        chai.add( chaiscript::fun( cross ), "cross");

        chai.add( chaiscript::fun( to_string ), "to_string");
        chai.add( chaiscript::fun( [](){Vector3 tmp; return std::move(tmp);} ), "Vector3");
        chai.add( chaiscript::fun( [](double a, double b, double c){Vector3 tmp = {a,b,c}; return std::move(tmp);} ), "Vector3");

        // Add geometry information
        chai.add(chaiscript::fun( [](std::array<int, 3> arr, int i){return arr[i];}), "[]");
        chai.add_global(chaiscript::const_var(&n_cells), "n_cells");
        chai.add_global(chaiscript::const_var(&n_cell_atoms), "n_cell_atoms");
        chai.add_global(chaiscript::const_var(&a_vec), "a_vec");
        chai.add_global(chaiscript::const_var(&b_vec), "b_vec");
        chai.add_global(chaiscript::const_var(&c_vec), "c_vec");
        chai.add_global(chaiscript::const_var(&center), "center");

        // Add spins and positions
        chai.add(chaiscript::fun( &Vector_View<Vector3>::operator[]), "[]"); // Normal index operator

        auto idx = [this] (const int a, const int b, const int c, const int i) { return i + this->n_cell_atoms * (a + this->n_cells[0] * (b + this->n_cells[1] * c));};
        chai.add(chaiscript::fun( idx ), "idx");

        chai.add(chaiscript::fun([&](const Vector_View<Vector3> & vec, const int a, const int b, const int c, const int i) {return vec[idx(a,b,c,i)]; }), "[]");
        chai.add(chaiscript::fun( &Vector_View<Vector3>::size), "size");
        chai.add_global(chaiscript::const_var(&positions), "positions");

        // We use the dynamic namespace to control access to variables that can change on an iteration basis
        chai.register_namespace(
            [this](chaiscript::Namespace& ns)
            {
                ns["spins"]    = chaiscript::const_var(&spins);
                ns["imported"] = chaiscript::const_var(true); // Use this to detect if the namespace was imported
            },
            "dynamic"
        );

        this->chai_state = chai.get_state();
    }

    void Spirit_Chai::update(State * p_state)
    {
        int nos   = Geometry_Get_NOS(p_state);
        spins     = Vector_View<Vector3>(reinterpret_cast<Vector3*>(System_Get_Spin_Directions(p_state)), nos);
        positions = Vector_View<Vector3>(reinterpret_cast<Vector3*>(Geometry_Get_Positions(p_state)), nos);

        int _n_cells[3];
        Geometry_Get_N_Cells(p_state, _n_cells);
        n_cells = { _n_cells[0], _n_cells[1], _n_cells[2] };

        float _a_vec[3], _b_vec[3], _c_vec[3];
        Geometry_Get_Bravais_Vectors(p_state, _a_vec, _b_vec, _c_vec);
        a_vec = { _a_vec[0], _a_vec[1], _a_vec[2] };
        b_vec = { _b_vec[0], _b_vec[1], _b_vec[2] };
        c_vec = { _c_vec[0], _c_vec[1], _c_vec[2] };

        float _center[3];
        Geometry_Get_Center(p_state, _center);
        center = { _center[0], _center[1], _center[2] };

        n_cell_atoms = Geometry_Get_N_Cell_Atoms(p_state);
    }

    void Spirit_Chai::reset_state()
    {
        chai.set_state(this->chai_state);
    }

    chaiscript::ChaiScript::State Spirit_Chai::state()
    {
        return chai.get_state();
    }

    void Spirit_Chai::set_state(const chaiscript::ChaiScript::State & state)
    {
        chai.set_state(state);
    }

    chaiscript::ChaiScript & Spirit_Chai::get()
    {
        return chai;
    }
}