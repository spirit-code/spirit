#include <fmm/Box.hpp>
#include <fmm/SimpleFMM_Defines.hpp>
#include <fmm/Tree.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>

int main()
{
    SimpleFMM::vectorfield pos;
    SimpleFMM::vectorfield spins;
    SimpleFMM::scalarfield mu_s;

    int Na           = 100;
    int Nb           = 100;
    int Nc           = 1;
    int N_ITERATIONS = 100;

    // build a simple cubic grid and some spins
    for( int a = 0; a < Na; a++ )
    {
        for( int b = 0; b < Nb; b++ )
        {
            for( int c = 0; c < Nc; c++ )
            {
                pos.push_back( { (double)a, (double)b, (double)c } );
                spins.push_back( { 0.0, 0.0, 1.0 } );
                mu_s.push_back( 1.0 );
            }
        }
    }

    // Build a tree
    std::chrono::high_resolution_clock::time_point t_pre  = std::chrono::high_resolution_clock::now();
    SimpleFMM::Tree tree                                  = SimpleFMM::Tree( 5, pos, 2, 5, 3 );
    std::chrono::high_resolution_clock::time_point t_post = std::chrono::high_resolution_clock::now();
    auto duration_setup                                   = ( t_post - t_pre ).count() / 1000000;
    std::cout << "Time for setup = " << duration_setup << " ms " << std::endl;

    SimpleFMM::vectorfield gradient( spins.size(), { 0, 0, 0 } );
    SimpleFMM::vectorfield gradient_direct( spins.size(), { 0, 0, 0 } );
    // for(auto it = tree.begin_level(0); it != tree.end_level(tree.n_level-1); it++)
    //     it->Print_Info(false, false);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    for( int i = 0; i < N_ITERATIONS; i++ )
    {
        tree.Upward_Pass( spins, mu_s );
        tree.Downward_Pass();
        tree.Evaluation( spins, mu_s, gradient );
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration                                     = ( t2 - t1 ).count() / 1000000;
    std::cout << "Duration " << N_ITERATIONS << " Iterations: " << duration << " ms" << std::endl;
    std::cout << "IPS = " << double( N_ITERATIONS ) / duration * 1000 << std::endl;
}