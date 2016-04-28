//
//  Class_DMRGSystem.hpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/26.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//


#ifndef Class_DMRGSystem_hpp
#define Class_DMRGSystem_hpp

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "Class_DMRGBlock.hpp"

using namespace Eigen;
using namespace std;

enum class SweepDirection {WR, L2R, R2L};
enum class FailSolution {RAND, TRUNC};

class DMRGSystem
{
public:
    MatrixXd c_up0;
    MatrixXd c_down0;
    MatrixXd u0;
    
    MatrixXd n_up0;
    MatrixXd n_down0;
    
    OperatorBlock H0;
    
    vector<DMRGBlock> BlockL;
    vector<DMRGBlock> BlockR;
    
    WavefunctionBlock psi;
    WavefunctionBlock seed;
    OperatorBlock rho;

    double u;   // Hubbard U
    
    double energy;
    double error;
    
    double rel_err;
    int max_lanczos_iter;
    
    int nsites;
    int d_per_site;
    int right_size;
    int left_size;
    
    SweepDirection state;
    FailSolution sol;
    bool fermion;
    
    DMRGSystem(int _nsites, int _max_lanczos_iter, double _rel_err, double u);
   
    void WarmUp(int total_QN, int n_states_to_keep, double truncation_error);
    void Sweep(int total_QN, int n_sweeps, int n_states_to_keep, double truncation_error);
    
    void BuildSeed(int n);
    void BuildBlock(BlockPosition _position);
    void GroundState(int site);
    double Truncate(BlockPosition _position, int _max_m, double _trun_err);
    
    void BuildOperator_n(BlockPosition pos, int site);
    void BuildOperator_c(BlockPosition pos, int site);
    void Measure();
};

double MeasureLocalDiag(const OperatorBlock& n, const WavefunctionBlock& psi, BlockPosition pos);
double MeasureTwoDiag(const OperatorBlock& op_left, const OperatorBlock& op_right, const WavefunctionBlock& psi);


#endif /* Class_DMRGSystem_hpp */
