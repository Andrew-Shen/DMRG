//
//  main.cpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/28.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//

//#define EIGEN_NO_AUTOMATIC_RESIZING
#define EIGEN_INITIALIZE_MATRICES_BY_NAN

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

#include "Class_DMRGSystem.hpp"

using namespace Eigen;
using namespace std;


int main()
{
    int nsites, n_sweeps, n_states_to_keep, max_lanczos_iter;
    double hubbard_u, particles;
    double rel_err, truncation_error;

    // DMRG Parameters
    nsites = 20;
    n_sweeps = 10;

    n_states_to_keep = 500;
    max_lanczos_iter = 500;
    truncation_error = 1e-7;
    rel_err = 1e-8;
    
    // Model Paramater
    hubbard_u = 1;
    particles = nsites - 4;
        
    DMRGSystem S(nsites, max_lanczos_iter, truncation_error, rel_err, hubbard_u);
    
    S.WarmUp(particles, n_states_to_keep, truncation_error * 10);

    S.Sweep(particles, n_sweeps, n_states_to_keep);

    return 0;
}