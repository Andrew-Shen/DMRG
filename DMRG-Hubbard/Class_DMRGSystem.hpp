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

enum class BlockPosition {LEFT,RIGHT};
enum class SweepDirection {WR, L2R, R2L};


class DMRGSystem
{
public:
    MatrixXd c_up0;
    MatrixXd c_down0;
    MatrixXd u0;
    
    MatrixXd sz0;
    MatrixXd n_up0;
    MatrixXd n_down0;
    
    vector<int> quantumN0;
    
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
    
    DMRGSystem(int _nsites, int _max_lanczos_iter, double _rel_err);
   
    void BuildBlockLeft(int _iter);
    void BuildBlockRight(int _iter);
    void GroundState(int n, bool wf_prediction);
    double Truncate(BlockPosition _position, int _max_m, double _trun_err);
    //void Measure();
    //void BuildSeed(SweepDirection dir);
};

/*
double measure_local(const MatrixXd &op, const MatrixXd &psi, BlockPosition pos);
double measure_two_site(const MatrixXd &op_left, const MatrixXd &op_right, const MatrixXd &psi);

*/
#endif /* Class_DMRGSystem_hpp */
