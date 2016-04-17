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

enum class WBType {RANDOM, ZERO, ONES};

WavefunctionBlock InitializeWavefunction(const vector<int>& quantumN_left, const vector<int>& quantumN_right,
                                       const vector<size_t>& block_size_left, const vector<size_t>& block_size_right,
                                       int n, WBType wb_type);
WavefunctionBlock InitializeWavefunction(const WavefunctionBlock& seed, WBType wb_type);
double InnerProd(WavefunctionBlock &v1, WavefunctionBlock &v2);



double Lanczos(DMRGSystem &S, int n, int _max_iter, double _rel_err);
WavefunctionBlock SuperBlockProd(DMRGSystem &S, WavefunctionBlock &psi);


double absl(double _a);

#endif /* Lanczos_hpp */
