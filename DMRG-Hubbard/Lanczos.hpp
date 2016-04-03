//
//  Lanczos.hpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/26.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//

#ifndef Lanczos_hpp
#define Lanczos_hpp

#include "Class_DMRGSystem.hpp"

#include <iostream>
#include <vector>
#include <Eigen/Core>

using namespace Eigen;
using namespace std;


WavefunctionBlock random_wavefunction(DMRGSystem &S, int n);
WavefunctionBlock zero_wavefunction(DMRGSystem &S, int n);
WavefunctionBlock one_wavefunction(DMRGSystem &S, int n);
double inner_product(WavefunctionBlock &v1, WavefunctionBlock &v2);



double Lanczos(DMRGSystem &S, int n, int _max_iter, double _rel_err, bool have_seed);
WavefunctionBlock symmetric_prod(DMRGSystem &S, int n, WavefunctionBlock &psi);


double absl(double _a);

#endif /* Lanczos_hpp */
